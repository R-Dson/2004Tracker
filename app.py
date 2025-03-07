import os
import json
from flask import Flask, render_template, jsonify, request
from data_fetcher import fetch_hiscores, compute_ehp, VALID_SKILLS
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
import re

def format_rate(value):
    """Format large numbers as 'k' format"""
    try:
        # Handle bytes-like objects
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        value = int(float(value))
    except (TypeError, ValueError):
        return str(value)
        
    if value >= 1000:
        return f"{value/1000:.0f}k"
    return str(value)

app = Flask(__name__)
app.config.from_object(Config)
app.jinja_env.filters['format_rate'] = format_rate

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
        
class SkillEHPLeaderboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(20), nullable=False)
    skill = db.Column(db.String(30), nullable=False)  # 'Overall' or skill name
    ehp = db.Column(db.Float, nullable=False)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Indexes and constraints
    __table_args__ = (
        db.Index('idx_skill_ehp', 'skill', 'ehp', 'username'),
        db.UniqueConstraint('user_id', 'skill', name='uix_user_skill')
    )
    
    user = db.relationship('User', backref='ehp_entries', lazy=True)

class SkillXPLeaderboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(20), nullable=False)
    skill = db.Column(db.String(30), nullable=False)
    xp_gain = db.Column(db.Integer, nullable=False)
    period_type = db.Column(db.String(10), nullable=False)  # daily/weekly/monthly
    period_start = db.Column(db.DateTime, nullable=False)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.Index('idx_skill_xp_period', 'skill', 'period_type', 'xp_gain', 'username'),
        db.UniqueConstraint('user_id', 'skill', 'period_type', name='uix_user_skill_period')
    )

    user = db.relationship('User', backref='xp_entries', lazy=True)

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

def ValidateUsername(username):
    """Validate username for security and format requirements"""
    if not username:
        return False, "Please provide username"
    
    # Check length (RuneScape usernames are limited to 12 characters)
    if len(username) > 13:
        return False, "Username must be 12 characters or less"
    
    # Check for valid characters (only letters, numbers, spaces and underscores allowed)
    if not re.match("^[a-zA-Z0-9_ ]*$", username):
        return False, "Username can only contain letters, numbers, spaces and underscores"
    
    return True, None

def check_rate_limit(username):
    """Check if username is rate limited. Returns (is_limited, timestamps)"""
    with rate_lock:
        now = datetime.utcnow().timestamp()
        timestamps = [ts for ts in rate_limits.get(username, []) if (now - ts) <= Config.RATE_LIMIT_WINDOW]
        rate_limits[username] = timestamps

    is_rate_limited = len(timestamps) >= 5
    
    if not is_rate_limited:
        timestamps.append(now)
        rate_limits[username] = timestamps
        
    return is_rate_limited, timestamps

def get_or_create_user(username):
    """Get existing user or create new one"""
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()
    return user

def update_user_data(username, user, data, force=False):
    """Update user data if changes detected. Returns (was_updated, error)"""
    if data is None:
        app.logger.error(f"Failed to fetch hiscores for user: {username}")
        return False, "Error retrieving hiscores"

    database_hiscores_data = HiscoresData.query.filter_by(user_id=user.id).order_by(HiscoresData.timestamp.asc()).all()
    print(force)
    if compare_hiscores_data(data, database_hiscores_data) or force:
        for item in data:
            hiscore_data = HiscoresData(
                skill=item['skill'],
                rank=item['rank'],
                level=item['level'],
                xp=item['xp'],
                author=user
            )
            db.session.add(hiscore_data)
        db.session.commit()

        # Update EHP leaderboard
        with open('data/skill_rates.json') as f:
            skill_rates = json.load(f)
        ehp_data, total_ehp = compute_ehp(data, skill_rates)
        update_ehp_leaderboard(username, user.id, ehp_data, total_ehp)
        return True, None

    return False, None

def validate_and_process_username(username):
    """Validate and process username. Returns (username, error_message)"""
    if not username:
        return None, "Please provide username"

    is_valid, error_message = ValidateUsername(username)
    if not is_valid:
        return None, error_message

    return username.strip().replace(' ', '_').lower(), None

def handle_update_request(username):
    """Handle update request for both API and web routes"""
    username, error = validate_and_process_username(username)
    if error:
        return None, error

    is_limited, timestamps = check_rate_limit(username)
    if is_limited:
        return None, "Rate limit exceeded"

    user = get_or_create_user(username)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing_today = HiscoresData.query.filter_by(user_id=user.id).filter(HiscoresData.timestamp >= today_start).first()
    force_update = existing_today is None  # Allow one update per day even without changes

    data = fetch_hiscores(username)
    was_updated, error = update_user_data(username, user, data, force=force_update)
    
    if error:
        return None, error
        
    return was_updated, None

@app.route('/api/update', methods=['POST'])
@rate_limit
def api_update():
    username = request.args.get('username')
    if username is None:
        app.logger.warning("API update request missing username")
        return jsonify({'error': 'Please provide username parameter'}), 400

    was_updated, error = handle_update_request(username)
    if error:
        return jsonify({'error': error}), 429 if error == "Rate limit exceeded" else 400

    return jsonify({
            'status': 'success',
            'updated': was_updated
        }), 200

def update_xp_leaderboard(username, progress_data, period_starts):
    """Update the XP leaderboard for daily, weekly, and monthly gains"""
    # Look up user by username
    user = User.query.filter_by(username=username).first()
    if not user:
        app.logger.error(f"User not found for XP leaderboard update: {username}")
        return

    for skill, data in progress_data.items():
        for period in ['daily', 'weekly', 'monthly']:
            xp_gain = data.get(f'{period}_xp')
            # Convert xp_gain to int if it exists
            if xp_gain:
                try:
                    xp_gain = int(xp_gain)
                    if xp_gain > 0:
                        process_xp_leaderboard_entry(skill, user.id, username, xp_gain, period, period_starts[period])
                except (TypeError, ValueError):
                    app.logger.error(f"Invalid XP value for {username} in {skill}: {xp_gain}")

def process_xp_leaderboard_entry(skill, user_id, username, xp_gain, period_type, period_start):
    """Process a single XP leaderboard entry, maintaining only top 10 entries per skill and period"""
    # Ensure xp_gain is an integer
    try:
        if isinstance(xp_gain, bytes):
            xp_gain = int(xp_gain.decode('utf-8'))
        else:
            xp_gain = int(float(xp_gain))
    except (TypeError, ValueError):
        app.logger.error(f"Invalid XP value in process_xp_leaderboard_entry: {xp_gain}")
        return

    # Get current top 10 for this skill and period
    current_top = SkillXPLeaderboard.query\
        .filter_by(skill=skill, period_type=period_type)\
        .order_by(SkillXPLeaderboard.xp_gain.desc())\
        .limit(10).all()
    
    # Check if user already has an entry
    existing = SkillXPLeaderboard.query.filter_by(
        user_id=user_id,
        skill=skill,
        period_type=period_type
    ).first()
    
    if len(current_top) < 10 or xp_gain > current_top[-1].xp_gain or existing:
        if existing:
            existing.xp_gain = xp_gain
            existing.period_start = period_start
            existing.last_updated = datetime.utcnow()
        else:
            new_entry = SkillXPLeaderboard(
                user_id=user_id,
                username=username,
                skill=skill,
                xp_gain=xp_gain,
                period_type=period_type,
                period_start=period_start
            )
            db.session.add(new_entry)
            
        # Remove lowest entry if we're over 10 and this is a new entry
        if len(current_top) >= 10 and not existing:
            lowest = current_top[-1]
            if lowest.xp_gain < xp_gain:  # Only remove if new entry has higher XP
                db.session.delete(lowest)
        
        db.session.commit()

@app.route('/fetch', methods=['GET', 'POST'])
@rate_limit
def fetch_hiscores_route():
    # Get username from either POST data or query params
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'username' not in data:
            app.logger.warning("Fetch POST request missing username")
            return jsonify({'error': 'Please provide username'}), 400
        username = data['username']
    else:
        username = request.args.get("username")

    username, error = validate_and_process_username(username)
    if error:
        app.logger.warning(f"Invalid username attempt: {username}")
        return jsonify({'error': error}), 400
    
    user = get_or_create_user(username)
    if not user:
        if request.method == 'GET':
            # For GET requests, just show empty data
            return render_template('fetch.html', 
                username=username,
                graphs=[],
                ehp_data={},
                total_ehp=0,
                progress_by_skill={},
                rank_progress={},
                hiscores_latest=[])

    database_hiscores_data = HiscoresData.query.filter_by(user_id=user.id).order_by(HiscoresData.timestamp.asc()).all()
    df = pd.DataFrame([(
        pd.to_datetime(item.timestamp),
        item.skill,
        item.rank,
        item.level,
        item.xp
    ) for item in database_hiscores_data], columns=['timestamp', 'skill', 'rank', 'level', 'xp']) if database_hiscores_data else pd.DataFrame(columns=['timestamp', 'skill', 'rank', 'level', 'xp'])

    if df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
 
    # Only check rate limits and fetch new data for POST requests
    if request.method == 'GET':
        return process_and_render_data(username, database_hiscores_data, df)
    
    was_updated, error = handle_update_request(username)
    if error:
        if error == "Rate limit exceeded":
            return process_and_render_data(username, database_hiscores_data, df)
        return jsonify({'error': error}), 400
    elif not was_updated:
        return process_and_render_data(username, database_hiscores_data, df)


    return process_and_render_data(username, database_hiscores_data, df)

def update_ehp_leaderboard(username, user_id, ehp_data, total_ehp):
    """Update the EHP leaderboard for a user's overall and skill-specific EHP"""
    # Update overall EHP
    process_leaderboard_entry('Overall', user_id, username, total_ehp)
    
    # Update individual skills
    for skill, ehp in ehp_data.items():
        process_leaderboard_entry(skill, user_id, username, ehp)

def process_leaderboard_entry(skill, user_id, username, ehp):
    """Process a single leaderboard entry, maintaining only top 10 entries per skill"""
    # Skip if EHP is 0 or less
    if ehp <= 1e-4:
        return
        
    # Check if user already has an entry
    existing = SkillEHPLeaderboard.query.filter_by(
        user_id=user_id,
        skill=skill
    ).first()
    
    # Get current top 10
    current_top = SkillEHPLeaderboard.query.filter_by(skill=skill)\
        .order_by(SkillEHPLeaderboard.ehp.desc())\
        .limit(10).all()
    
    if len(current_top) < 10 or ehp > current_top[-1].ehp or existing:
        if existing:
            existing.ehp = ehp
            existing.last_updated = datetime.utcnow()
        else:
            new_entry = SkillEHPLeaderboard(
                user_id=user_id,
                username=username,
                skill=skill,
                ehp=ehp
            )
            db.session.add(new_entry)
            
        # Remove lowest entry if we're over 10 and this is a new entry
        if len(current_top) >= 10 and not existing:
            lowest = current_top[-1]
            if lowest.ehp < ehp:  # Only remove if new entry has higher EHP
                db.session.delete(lowest)
        
        db.session.commit()

@app.route('/ehp-leaderboard')
def ehp_leaderboard():
    skills = ['Overall'] + sorted([s for s in VALID_SKILLS if s != 'Overall'])
    leaderboards = {}

    # First get Overall data
    leaders = SkillEHPLeaderboard.query\
        .filter_by(skill='Overall')\
        .order_by(SkillEHPLeaderboard.ehp.desc())\
        .limit(10)\
        .all()
    leaderboards['Overall'] = leaders

    # Then get other skills in alphabetical order
    for skill in sorted([s for s in skills if s != 'Overall']):
        # Skip unwanted skills
        if skill in ['Farming', 'Construction', 'Hunter']:
            continue
            
        leaders = SkillEHPLeaderboard.query\
            .filter_by(skill=skill)\
            .order_by(SkillEHPLeaderboard.ehp.desc())\
            .limit(10)\
            .all()
        leaderboards[skill] = leaders
    
    return render_template('ehp_leaderboard.html', leaderboards=leaderboards)

@app.route('/records')
def records():
    """Display XP gain leaderboards for different time periods"""
    filtered_skills = [s for s in VALID_SKILLS if s != 'Overall' and s not in ['Farming', 'Construction', 'Hunter']]
    skills = ['Overall'] + sorted(filtered_skills)
    periods = ['daily', 'weekly', 'monthly']
    leaderboards = {}
    
    for period in periods:
        period_boards = {}
        for skill in skills:
            leaders = SkillXPLeaderboard.query\
                .filter_by(skill=skill, period_type=period)\
                .order_by(SkillXPLeaderboard.xp_gain.desc())\
                .limit(10)\
                .all()
            if leaders:  # Only add if there are entries
                period_boards[skill] = leaders
        leaderboards[period] = period_boards
    
    return render_template('records.html', 
                         leaderboards=leaderboards,
                         periods=periods,
                         skills=skills)

def process_and_render_data(username, database_hiscores_data, df):
    """Helper function to process and render the hiscores data"""
    df_overall = df[df['skill'] == 'Overall'].copy()
    df_overall = df_overall.sort_values('timestamp')
    df_skills = df[df['skill'] != 'Overall'].copy()
    df_skills = df_skills[df_skills['level'] > 1]
    
    # Filter out unwanted skills
    df_skills = df_skills[
        ~df_skills['skill'].isin(['Farming', 'Construction', 'Hunter'])
    ]
    
    df_skills = df_skills.sort_values(['skill', 'timestamp'])
    
    # Get first occurrence of each level for each skill
    df_first_levels = df_skills.loc[
        df_skills.groupby(['skill', 'level'])['timestamp'].idxmin()
    ]

    # Get the most recent points
    df_latest = df_skills.loc[df_skills.groupby('skill')['timestamp'].idxmax()]

    df_skills = pd.concat([df_first_levels, df_latest]).drop_duplicates(['skill', 'timestamp']).reset_index(drop=True)    
    df_skills = df_skills.sort_values(['skill', 'timestamp'])

    graphs = []

    if not df_overall.empty:
        # For the overall level graph
        fig_overall = px.line(
            df_overall, 
            x='timestamp', 
            y='level', 
            markers=True,
            title=f'Total Level Progression for {username}',
            labels={'timestamp': 'Date', 'level': 'Level'},
            color_discrete_sequence=['#FF6B6B']  # A nicer red color
        )

        # Update y-axis settings
        fig_overall.update_yaxes(
            rangemode="nonnegative", 
            dtick=10, 
            tickformat='d',
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.5)',
            linewidth=0.5,
            title_font=dict(size=14)
        )

        # Update x-axis settings
        fig_overall.update_xaxes(
            tickformat="%b %d\n%Y",
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.5)',
            linewidth=0.5,
            title_font=dict(size=14)
        )

        # Update traces (lines and markers)
        fig_overall.update_traces(
            marker=dict(size=6),
            line=dict(width=2),
            hovertemplate='<b>Total Level</b><br>Date: %{x|%b %d, %Y}<br>Level: %{y}<extra></extra>'
        )

        # Update overall layout
        fig_overall.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=30, r=20, t=60, b=100),
            font_color='#F0F0F0',
            font_family="Arial",
            title=dict(
                text=f'Total Level Progression for {username}',
                font=dict(size=18),
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                orientation="h",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0.5
            ),
            hoverlabel=dict(
                bgcolor="#211C15",
                font_size=16,
                font_family="Arial",
                font_color="white",
                bordercolor="#555"
            )
        )

        # Add to graphs list
        graphs.append(plotly.offline.plot(fig_overall, output_type='div'))

        # For the skills graph (if data exists)
        if not df_skills.empty:
            # Create skills figure
            fig_skills = px.line(
                df_skills, 
                x='timestamp', 
                y='level', 
                color='skill', 
                markers=True,
                title=f'Skill Level Progression for {username}',
                labels={'timestamp': 'Date', 'level': 'Level', 'skill': 'Skill'},
                color_discrete_sequence=px.colors.qualitative.Vivid  # More vibrant color scheme
            )
            
            # Update y-axis settings
            fig_skills.update_yaxes(
                rangemode="nonnegative", 
                dtick=10, 
                tickformat='d',
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.5)',
                linewidth=0.5,
                title_font=dict(size=14)
            )
            
            # Update x-axis settings
            fig_skills.update_xaxes(
                tickformat="%b %d\n%Y",
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.5)',
                linewidth=0.5,
                title_font=dict(size=14)
            )
            
            # Update traces (lines and markers)
            fig_skills.update_traces(
                marker=dict(size=6),
                line=dict(width=1.5),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %d, %Y}<br>Level: %{y}<extra></extra>'
            )
            
            # Update overall layout
            fig_skills.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600,
                margin=dict(l=30, r=20, t=60, b=100),
                font_color='#F0F0F0',
                font_family="Arial",
                title=dict(
                    text=f'Skill Level Progression for {username}',
                    font=dict(size=18),
                    x=0.5,
                    xanchor='center'
                ),
                legend=dict(
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0.5
                ),
                hoverlabel=dict(
                    bgcolor="#211C15",
                    font_size=16,
                    font_family="Arial",
                    font_color="white",
                    bordercolor="#555"
                )
            )
            
            # Add to graphs list
            fig_skills.update_xaxes(tickformat="%Y-%m-%d %H:00")
            graphs.append(plotly.offline.plot(fig_skills, output_type='div'))

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
        # Skip unwanted skills
        if skill in ['Farming', 'Construction', 'Hunter']:
            continue
            
        if skill != 'Overall':
            hiscores_latest.append({
                "skill": latest_records[skill].skill,
                "xp": latest_records[skill].xp,
                "rank": latest_records[skill].rank,
                "level": latest_records[skill].level
            })

    # Filter out unwanted skills before computing EHP
    hiscores_latest = [item for item in hiscores_latest if item['skill'] not in ['Farming', 'Construction', 'Hunter']]
    ehp_data, total_ehp = compute_ehp(hiscores_latest, skill_rates)

    from datetime import timedelta
    now = datetime.utcnow()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - timedelta(days=1)
    week_boundary = now - timedelta(days=7)
    month_boundary = now - timedelta(days=30)
    
    level_progress = compute_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now)
    rank_progress = compute_rank_progress_by_skill(df, today_midnight, yesterday_midnight, week_boundary, month_boundary, now)
    
    # Update XP leaderboard
    update_xp_leaderboard(username, level_progress, {
        'daily': yesterday_midnight,
        'weekly': week_boundary,
        'monthly': month_boundary
    })
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
    skills = [s for s in skills if s not in ['Farming', 'Construction', 'Hunter']]

    # Get the earliest timestamp for each skill
    earliest_timestamps = {}
    for skill in skills:
        skill_data = df[df['skill'] == skill]
        if not skill_data.empty:
            earliest_timestamps[skill] = skill_data['timestamp'].min()
    
    if 'Overall' in skills:
        # Remove and add Overall first
        skills.remove('Overall')
        skills = ['Overall'] + skills

    for skill in skills:
        earliest_timestamp = earliest_timestamps.get(skill)
        if earliest_timestamp is None:
            continue
            
        # Get data for this skill sorted by timestamp
        df_skill = df[df['skill'] == skill].sort_values('timestamp')
        
        # Daily progress - find first non-zero XP in the period
        # Get all data points from today
        today = now.date()
        df_daily_skill = df_skill[df_skill['timestamp'].dt.date == today]
        daily = None
        daily_xp = None
        
        # Skip if no data points in period or only one data point (need at least 2 for gain calculation)
        if len(df_daily_skill) <= 1:
            daily = 0
            daily_xp = 0
        else:
            # Use first and last data points from today
            first_today = df_daily_skill.iloc[0]
            last_today = df_daily_skill.iloc[-1]
            daily = last_today['level'] - first_today['level']
            daily_xp = last_today['xp'] - first_today['xp']
        
        # Weekly progress - find first non-zero XP in the period
        df_weekly_skill = df_skill[(df_skill['timestamp'] >= week_boundary) & (df_skill['timestamp'] <= now)]
        weekly = None
        weekly_xp = None

        if len(df_weekly_skill) <= 1:
            weekly = 0
            weekly_xp = 0
        else:
            # Use first and last data points from the week
            first_week = df_weekly_skill.iloc[0]
            last_week = df_weekly_skill.iloc[-1]
            weekly = last_week['level'] - first_week['level']
            weekly_xp = last_week['xp'] - first_week['xp']
        
        # Monthly progress - find first non-zero XP in the period
        df_monthly_skill = df_skill[(df_skill['timestamp'] >= month_boundary) & (df_skill['timestamp'] <= now)]
        monthly = None
        monthly_xp = None
        
        if len(df_monthly_skill) <= 1:
            monthly = 0
            monthly_xp = 0
        else:
            first_month = df_monthly_skill.iloc[0]
            last_month = df_monthly_skill.iloc[-1]
            monthly = last_month['level'] - first_month['level']
            monthly_xp = last_month['xp'] - first_month['xp']
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

