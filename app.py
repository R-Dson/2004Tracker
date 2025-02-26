from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.offline
import math
import time

app = Flask(__name__)
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

@app.route('/fetch', methods=['GET'])
def fetch_hiscores_route():
    from flask import request
    from data_fetcher import fetch_hiscores

    username = request.args.get("username")
    if not username:
        return "Please provide username", 400
    data = fetch_hiscores(username)
    if data is None:
        return "Error retrieving hiscores", 500

    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)

    existing_hiscores_data_list = HiscoresData.query.filter_by(user_id=user.id).order_by(HiscoresData.timestamp.desc()).all()

    if not compare_hiscores_data(data, existing_hiscores_data_list):
        pass # No updates, do not store
    else:
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

    # Retrieve data from database to display (always)
    database_hiscores_data = HiscoresData.query.filter_by(user_id=user.id).order_by(HiscoresData.timestamp.asc()).all()
    if not database_hiscores_data:
        return f"<h2>Hiscores for {username} (Database)</h2><p>No data found in database for {username}.</p>"
    
    df = pd.DataFrame([(
        item.timestamp,
        item.skill,
        item.level,
        item.xp
    ) for item in database_hiscores_data], columns=['timestamp', 'skill', 'level', 'xp'])

    if df.empty:
        return render_template('index.html', username=username, graphs=None)

    df_overall = df[df['skill'] == 'Overall'].copy()
    df_overall = df_overall.sort_values('timestamp')
    df_skills = df[df['skill'] != 'Overall'].copy()
    df_skills = df_skills.groupby(['skill', pd.Grouper(key='timestamp', freq='H')]).last().reset_index()
    df_skills = df_skills[df_skills['level'] > 1]

    graphs = []

    if not df_overall.empty:
        fig_overall = px.line(df_overall, x='timestamp', y='level', markers=True,
                              title=f'Overall Level Progression for {username}',
                              labels={'timestamp': 'Date', 'level': 'Level'})
        fig_overall.update_yaxes(rangemode="nonnegative", dtick=10, tickformat='d')
        fig_overall.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#F0F0F0') #, grid={'color': '#555'})
        graphs.append(plotly.offline.plot(fig_overall, output_type='div'))
        fig_overall.update_xaxes(tickformat="%Y-%m-%d %H:00")
    if not df_skills.empty:
        fig_skills = px.line(df_skills, x='timestamp', y='level', color='skill', markers=True,
                             # color_discrete_sequence=px.colors.qualitative.Dark24,
                             title=f'Skill Level Progression for {username}',
                             labels={'timestamp': 'Date', 'level': 'Level'})
        fig_skills.update_yaxes(rangemode="nonnegative", dtick=10, tickformat='d')
        fig_skills.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#F0F0F0') #, grid={'color': '#555'})
        graphs.append(plotly.offline.plot(fig_skills, output_type='div'))
        fig_skills.update_xaxes(tickformat="%Y-%m-%d %H:00")

    # Compute efficient hours played using latest record per skill
    import json
    with open('data/skill_rates.json') as f:
        skill_rates = json.load(f)
    latest_records = {}
    for record in database_hiscores_data:
        if record.skill not in latest_records or record.timestamp > latest_records[record.skill].timestamp:
            latest_records[record.skill] = record
    hiscores_latest = []
    for skill in latest_records:
        hiscores_latest.append({
            "skill": latest_records[skill].skill,
            "xp": latest_records[skill].xp
        })
    from data_fetcher import compute_ehp
    ehp_data, total_ehp = compute_ehp(hiscores_latest, skill_rates)

    # Compute progress data per skill
    from datetime import timedelta
    now = datetime.utcnow()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - timedelta(days=1)
    week_boundary = now - timedelta(days=7)
    month_boundary = now - timedelta(days=30)

    progress_by_skill = {}
    for skill in df['skill'].unique():
        df_skill = df[df['skill'] == skill].sort_values('timestamp')
        # Daily progress for this skill
        df_daily_skill = df_skill[(df_skill['timestamp'] >= yesterday_midnight) & (df_skill['timestamp'] < today_midnight)]
        daily = None
        if not df_daily_skill.empty:
            daily = df_daily_skill.iloc[-1]['level'] - df_daily_skill.iloc[0]['level']
        # Weekly progress for this skill
        df_weekly_skill = df_skill[(df_skill['timestamp'] >= week_boundary) & (df_skill['timestamp'] <= now)]
        weekly = None
        if not df_weekly_skill.empty:
            weekly = df_weekly_skill.iloc[-1]['level'] - df_weekly_skill.iloc[0]['level']
        # Monthly progress for this skill
        df_monthly_skill = df_skill[(df_skill['timestamp'] >= month_boundary) & (df_skill['timestamp'] <= now)]
        monthly = None
        if not df_monthly_skill.empty:
            monthly = df_monthly_skill.iloc[-1]['level'] - df_monthly_skill.iloc[0]['level']
        progress_by_skill[skill] = {"daily": daily, "weekly": weekly, "monthly": monthly}

    return render_template('fetch.html', username=username, graphs=graphs, ehp_data=ehp_data, total_ehp=total_ehp, progress_by_skill=progress_by_skill)


def compare_hiscores_data(fetched_data, existing_data_list):
    """
    Compares fetched hiscores data with existing hiscores data from the database,
    only updating if the overall level has increased.
    Returns True if update is needed, False otherwise.
    """
    if not existing_data_list:
        return True  # No existing data, so fetched data is always different

    fetched_overall_level = None
    existing_overall_level = None

    for item in fetched_data:
        if item['skill'] == 'Overall':
            fetched_overall_level = item['level']
            break

    for item in existing_data_list:
        if item.skill == 'Overall':
            existing_overall_level = item.level
            break

    if fetched_overall_level is None:
        return False # Should not happen, but handle just in case
    
    if existing_overall_level is None:
        return True # No existing overall level, so update

    return fetched_overall_level > existing_overall_level

@app.route('/rates')
def rates():
    import json
    with open('data/skill_rates.json') as f:
        skill_rates = json.load(f)
    return render_template('rates.html', skill_rates=skill_rates)

    

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
