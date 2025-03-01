import os
import json
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.offline
from threading import Lock
import logging.handlers
import logging
from config import Config
import functools
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.config.from_object(Config)

# Configure proxy fix
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'"
    return response

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

file_handler = logging.handlers.RotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.setLevel(getattr(logging, Config.LOG_LEVEL))

# Error handler for 500 errors
@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    hiscores_data = db.relationship('HiscoresData', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}')"

class HiscoresData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    skill = db.Column(db.String(30), nullable=False)
    rank = db.Column(db.Integer)
    level = db.Column(db.Integer)
    xp = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"HiscoresData('{self.timestamp}', '{self.skill}', '{self.rank}', '{self.level}', '{self.xp}')"

# Rate limiting globals
rate_limits = {}
rate_lock = Lock()

def rate_limit(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Move rate limiting logic here
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def home():
    recently_updated_users = get_recently_updated_users()
    return render_template('index.html', recently_updated_users=recently_updated_users)

def get_recently_updated_users():
    """
    Fetches a list of recently updated users, ordered by the timestamp of their latest hiscore data.
    """
    users = User.query.join(HiscoresData).group_by(User.id).order_by(db.desc(db.func.max(HiscoresData.timestamp))).limit(10).all() # Limit to 10 for recently updated
    return users

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/fetch', methods=['GET'])
@rate_limit
def fetch_hiscores_route():
    from flask import request
    from data_fetcher import fetch_hiscores

    username = request.args.get("username")
    if not username:
        app.logger.warning("Fetch request missing username")
        return jsonify({'error': 'Please provide username'}), 400
        
    # Get existing data first
    with rate_lock:
        now = datetime.utcnow().timestamp()
        timestamps = [ts for ts in rate_limits.get(username, [])
                     if (now - ts) <= Config.RATE_LIMIT_WINDOW]
        rate_limits[username] = timestamps  # Update the list of valid timestamps
        
    # If we have existing data and are rate limited, return it immediately
    is_rate_limited = len(timestamps) >= 5
    user = User.query.filter_by(username=username).first()
    if user:
        database_hiscores_data = HiscoresData.query.filter_by(user_id=user.id).order_by(HiscoresData.timestamp.asc()).all()
        if database_hiscores_data:
            df = pd.DataFrame([(
                pd.to_datetime(item.timestamp),
                item.skill,
                item.rank,
                item.level,
                item.xp
            ) for item in database_hiscores_data], columns=['timestamp', 'skill', 'rank', 'level', 'xp'])
            
            if not df.empty:
                # If rate limited, return existing data
                if is_rate_limited:
                    return process_and_render_data(username, database_hiscores_data, df)
    else:
        user = User(username=username)
        db.session.add(user)
        database_hiscores_data = []
        df = pd.DataFrame(columns=['timestamp', 'skill', 'rank', 'level', 'xp'])
        # Ensure timestamp column is datetime type even for empty DataFrame
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Rate limiting check
    with rate_lock:
        if is_rate_limited:
                
            return process_and_render_data(username, database_hiscores_data, df)
        else:
            timestamps.append(now)
            rate_limits[username] = timestamps
            print(rate_limits[username])

            data = fetch_hiscores(username)
            if data is None:
                app.logger.error(f"Failed to fetch hiscores for user: {username}")
                return jsonify({'error': 'Error retrieving hiscores'}), 500

    # Continue with normal flow if not rate limited
    if data is None:
        app.logger.error(f"Failed to fetch hiscores for user: {username}")
        return jsonify({'error': 'Error retrieving hiscores'}), 500

    if compare_hiscores_data(data, database_hiscores_data):
        for item in data:
            hiscore_data = HiscoresData(
                skill=item['skill'],
                rank=item['rank'],
                level=item['level'],
                xp=item['xp'],
                author=user
            )
            db.session.add(hiscore_data)
        app.logger.info(f"Updated hiscores data for user: {username}")
        db.session.commit()

    return process_and_render_data(username, database_hiscores_data, df)

def process_and_render_data(username, database_hiscores_data, df):
    """Helper function to process and render the hiscores data"""
    df_overall = df[df['skill'] == 'Overall'].copy()
    df_overall = df_overall.sort_values('timestamp')
    df_skills = df[df['skill'] != 'Overall'].copy()
    df_skills = df_skills.groupby(['skill', pd.Grouper(key='timestamp', freq='H')]).last().reset_index()
    df_skills = df_skills[df_skills['level'] > 1]

    graphs = []

    if not df_overall.empty:
        fig_overall = px.line(df_overall, x='timestamp', y='level', markers=True,
                           title=f'Total Level Progression for {username}',
                           labels={'timestamp': 'Date', 'level': 'Level'})
        fig_overall.update_yaxes(rangemode="nonnegative", dtick=10, tickformat='d')
        fig_overall.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            margin=dict(l=30, r=10, t=50, b=100),
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True),
            font_color='#F0F0F0',
            legend=dict(
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5,
                orientation="h"
            )
        )
        graphs.append(plotly.offline.plot(fig_overall, output_type='div'))
        fig_overall.update_xaxes(tickformat="%Y-%m-%d %H:00")
    if not df_skills.empty:
        fig_skills = px.line(df_skills, x='timestamp', y='level', color='skill', markers=True,
                          title=f'Skill Level Progression for {username}',
                          labels={'timestamp': 'Date', 'level': 'Level'})
        fig_skills.update_yaxes(rangemode="nonnegative", dtick=10, tickformat='d')
        fig_skills.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            margin=dict(l=30, r=10, t=50, b=100),
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True),
            font_color='#F0F0F0',
            legend=dict(
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5,
                orientation="h"
            )
        )
        graphs.append(plotly.offline.plot(fig_skills, output_type='div'))
        fig_skills.update_xaxes(tickformat="%Y-%m-%d %H:00")

    # Compute progress data per skill
    import json
    with open('data/skill_rates.json') as f:
        skill_rates = json.load(f)
    latest_records = {}
    for record in database_hiscores_data:
        if record.skill not in latest_records or record.timestamp > latest_records[record.skill].timestamp:
            latest_records[record.skill] = record
    hiscores_latest = []
    # First add Overall if it exists
    if 'Overall' in latest_records:
        hiscores_latest.append({
            "skill": latest_records['Overall'].skill,
            "xp": latest_records['Overall'].xp,
            "rank": latest_records['Overall'].rank,
            "level": latest_records['Overall'].level
        })
    # Then add all other skills
    for skill in sorted(latest_records):
        if skill != 'Overall':
            hiscores_latest.append({
                "skill": latest_records[skill].skill,
                "xp": latest_records[skill].xp,
                "rank": latest_records[skill].rank,
                "level": latest_records[skill].level
            })
    from data_fetcher import compute_ehp
    ehp_data, total_ehp = compute_ehp(hiscores_latest, skill_rates)

    from datetime import timedelta
    now = datetime.utcnow()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - timedelta(days=1)
    week_boundary = now - timedelta(days=7)
    month_boundary = now - timedelta(days=30)
    
    level_progress = compute_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now)
    rank_progress = compute_rank_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now)
    
    return render_template('fetch.html', username=username, graphs=graphs, ehp_data=ehp_data, total_ehp=total_ehp,
                         progress_by_skill=level_progress, rank_progress=rank_progress, hiscores_latest=hiscores_latest)

def compare_hiscores_data(fetched_data, existing_data_list):
    """
        Compares fetched hiscores data with existing hiscores data from the database,
        updating if either:
        1. The overall experience has increased
        2. Any skill rank has changed
        Returns True if update is needed, False otherwise.
        """
    if not existing_data_list:
        return True  # No existing data, so fetched data is always different

    fetched_overall_xp = None
    existing_overall_xp = None

    for item in fetched_data:
        if item['skill'] == 'Overall':
            fetched_overall_xp = item['xp']
            break

    for item in existing_data_list:
        if item.skill == 'Overall':
            existing_overall_xp = item.xp
            break

    if fetched_overall_xp is None:
        return False # Should not happen, but handle just in case
    
    if existing_overall_xp is None:
        return True # No existing overall xp, so update

    # Check if overall XP increased
    if fetched_overall_xp > existing_overall_xp:
        return True

    # Check if any ranks changed
    for fetched_item in fetched_data:
        for existing_item in existing_data_list:
            if fetched_item['skill'] == existing_item.skill:
                if fetched_item['rank'] != existing_item.rank:
                    return True

    return False

def compute_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now):
    progress_by_skill = {}
    skills = sorted(df['skill'].unique())
    if 'Overall' in skills:
        # Remove and add Overall first
        skills.remove('Overall')
        skills = ['Overall'] + skills

    for skill in skills:
        # Get data for this skill sorted by timestamp
        df_skill = df[df['skill'] == skill].sort_values('timestamp')
        # Daily progress for this skill
        df_daily_skill = df_skill[(df_skill['timestamp'] >= yesterday_midnight) & (df_skill['timestamp'] < today_midnight)]
        daily = None
        daily_xp = None
        if not df_daily_skill.empty:
            daily = df_daily_skill.iloc[-1]['level'] - df_daily_skill.iloc[0]['level']
            daily_xp = df_daily_skill.iloc[-1]['xp'] - df_daily_skill.iloc[0]['xp']
        # Weekly progress for this skill
        df_weekly_skill = df_skill[(df_skill['timestamp'] >= week_boundary) & (df_skill['timestamp'] <= now)]
        weekly = None
        weekly_xp = None
        if not df_weekly_skill.empty:
            weekly = df_weekly_skill.iloc[-1]['level'] - df_weekly_skill.iloc[0]['level']
            weekly_xp = df_weekly_skill.iloc[-1]['xp'] - df_weekly_skill.iloc[0]['xp']
        # Monthly progress for this skill
        df_monthly_skill = df_skill[(df_skill['timestamp'] >= month_boundary) & (df_skill['timestamp'] <= now)]
        monthly = None
        monthly_xp = None
        if not df_monthly_skill.empty:
            monthly = df_monthly_skill.iloc[-1]['level'] - df_monthly_skill.iloc[0]['level']
            monthly_xp = df_monthly_skill.iloc[-1]['xp'] - df_monthly_skill.iloc[0]['xp']
        progress_by_skill[skill] = {
            "daily": daily, "weekly": weekly, "monthly": monthly,
            "daily_xp": daily_xp, "weekly_xp": weekly_xp, "monthly_xp": monthly_xp
        }
    return progress_by_skill

def compute_rank_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now):
    """
    Compute rank progress for each skill over different time periods.
    A negative change in rank means improvement (moving up in rankings).
    """
    rank_progress = {}
    for skill in df['skill'].unique():
        df_skill = df[df['skill'] == skill].sort_values('timestamp')
        # Daily progress
        df_daily_skill = df_skill[(df_skill['timestamp'] >= yesterday_midnight) & (df_skill['timestamp'] < today_midnight)]
        daily = None
        if not df_daily_skill.empty:
            daily = df_daily_skill.iloc[0]['rank'] - df_daily_skill.iloc[-1]['rank']
            
        # Weekly progress
        df_weekly_skill = df_skill[(df_skill['timestamp'] >= week_boundary) & (df_skill['timestamp'] <= now)]
        weekly = None
        if not df_weekly_skill.empty:
            weekly = df_weekly_skill.iloc[0]['rank'] - df_weekly_skill.iloc[-1]['rank']
            
        # Monthly progress
        df_monthly_skill = df_skill[(df_skill['timestamp'] >= month_boundary) & (df_skill['timestamp'] <= now)]
        monthly = None
        if not df_monthly_skill.empty:
            monthly = df_monthly_skill.iloc[0]['rank'] - df_monthly_skill.iloc[-1]['rank']
        rank_progress[skill] = {"daily": daily, "weekly": weekly, "monthly": monthly}
    return rank_progress




@app.route('/rates')
def rates():
    import json
    with open('data/skill_rates.json') as f:
        skill_rates = json.load(f)
    return render_template('rates.html', skill_rates=skill_rates)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # export FLASK_ENV=development
    if app.config['ENV'] == 'development':
        app.run(host='0.0.0.0', port=5002, debug=True)
    else:
        # In production, use Gunicorn
        app.logger.info('Production mode - use Gunicorn to run the application')
        exit(1)

