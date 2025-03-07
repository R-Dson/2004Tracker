import requests
from bs4 import BeautifulSoup
from utils.exp import get_xp_for_level

def fetch_hiscores(username):
    """
    Fetch the hiscores page for the given username, parse the HTML to locate the hiscores table,
    and extract the data as a list of dictionaries.
    
    Each dictionary contains:
        - skill: Skill name
        - rank: Rank (integer if possible)
        - level: Level (integer if possible)
        - xp: XP (integer if possible)
    Returns None if an error occurs.
    """
    url = f"https://2004.lostcity.rs/hiscores/player/{username}"
    print(f"Fetching URL: {url}")
    try:
        response = requests.get(url)
        print(f"Response code: {response.status_code}")
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching hiscores for {username}: {e}")
        return None

    # Save HTML for debugging
    with open(f"{username}_debug.html", "w") as f:
        f.write(response.text)

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Try multiple ways to locate the hiscores table
    # First try: Look for table with bgcolor="black"
    outer_table = soup.find('table', attrs={'bgcolor': 'black'})
    if outer_table:
        inner_table = outer_table.find('table')
        if inner_table:
            return parse_hiscores_table(inner_table)
    
    # Second try: Look for any table with "Hiscores" in its text
    tables = soup.find_all('table')
    for table in tables:
        if "Hiscores" in table.get_text():
            return parse_hiscores_table(table)
    
    print(f"Could not locate hiscores table for {username}.")
    return None

VALID_SKILLS = [
    "Overall", "Attack", "Strength", "Defence", "Hitpoints", "Ranged", 
    "Magic", "Cooking", "Mining", "Crafting", "Prayer", "Runecrafting", 
    "Firemaking", "Smithing", "Fishing", "Thieving", "Fletching", 
    "Woodcutting", "Herblore", "Agility"
] # , "Hunter", "Construction", "Farming", "Slayer" # Not in the game yet

def parse_hiscores_table(table):
    """Parse the hiscores table and return structured data"""
    rows = table.find_all('tr')
    if not rows or len(rows) < 2:
        print("Not enough rows found in the hiscores table.")
        return None

    # Skip the first three rows (header, empty row, and title row)
    data_rows = rows[3:]
    
    # Create a dictionary to store found skills
    found_skills = {}
    for row in data_rows:
        cols = row.find_all('td')
        # Expecting at least 6 columns: first two might be empty, then skill, rank, level, xp.
        if len(cols) >= 6:
            try:
                skill = cols[2].get_text(strip=True)
                # Only process if it's a valid skill
                if skill in VALID_SKILLS:
                    rank_text = cols[3].get_text(strip=True).replace(",", "")
                    level_text = cols[4].get_text(strip=True).replace(",", "")
                    xp_text = cols[5].get_text(strip=True).replace(",", "")
                    rank = int(rank_text) if rank_text.isdigit() else rank_text
                    level = int(level_text) if level_text.isdigit() else level_text
                    xp = int(xp_text) if xp_text.isdigit() else xp_text
                    found_skills[skill] = {
                        "skill": skill,
                        "rank": rank,
                        "level": level,
                        "xp": xp
                    }
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
    
    # Create complete hiscores list with default values for missing skills
    hiscores = []
    for skill in VALID_SKILLS:
        if skill in found_skills:
            hiscores.append(found_skills[skill])
        else:
            hiscores.append({
                "skill": skill,
                "rank": 0,
                "level": 1,
                "xp": 0
            })
    
    return hiscores


def compute_ehp(hiscores, skill_rates):
    """
    Compute the efficient hours played (EHP) for each skill based on current XP and skill rates.
    EHP for a skill is calculated by distributing the current XP across the defined intervals,
    subtracting any EHP earned through bonus XP to get the real direct training EHP needed.
    dividing the XP in each interval by its rate, and summing the results.
    Returns a tuple: (ehp_data, total_ehp)
    """
    ehp_data = {}
    total_ehp = 0
    for entry in hiscores:
        skill = entry["skill"]
        if skill == "Overall": continue
        xp = entry["xp"]
        if skill not in skill_rates["skills"]:
            continue
            
        # First calculate bonus XP this skill gets while training other skills
        bonus_xp = 0
        bonus_time = 0
        for other_skill, intervals in skill_rates["skills"].items():
            if other_skill == skill or other_skill == "Overall":
                continue
                
            # Find the other skill's current XP
            other_entry = next((h for h in hiscores if h["skill"] == other_skill), None)
            if not other_entry:
                continue
                
            # Check each training interval for bonus XP
            other_xp = other_entry["xp"]
            for interval in intervals:
                if "bonus_skills" not in interval:
                    continue
                    
                for bonus in interval["bonus_skills"]:
                    if bonus["skill"] == skill:
                        training_time = min(other_xp, interval["rate"]) / interval["rate"]
                        bonus_xp += training_time * bonus["rate"]
                        bonus_time += training_time
                        
        intervals = skill_rates["skills"][skill]
        skill_ehp = 0  # Will calculate total direct training time
        remaining_xp = xp

        for interval in intervals:
            min_level = interval["min_level"]
            max_level = interval["max_level"]
            rate = interval["rate"]

            interval_max_level = max_level if max_level is not None else 99
            interval_start_level = max(min_level, 1)  # Ensure start level is at least 1

            start_xp = 0
            if interval_start_level > 1:
                start_xp = get_xp_for_level(interval_start_level - 1)

            end_xp = get_xp_for_level(interval_max_level)

            if remaining_xp <= 0:
                break

            interval_range = end_xp - start_xp
            xp_in_interval = min(remaining_xp, interval_range)
            interval_ehp = xp_in_interval / rate
            skill_ehp += interval_ehp
            remaining_xp -= xp_in_interval

        # Subtract bonus time to get real EHP needed
        real_ehp = max(0, skill_ehp - bonus_time)  # Ensure we don't go negative

        ehp_data[skill] = round(real_ehp, 1)
        total_ehp += real_ehp

    return ehp_data, round(total_ehp, 1)

if __name__ == "__main__":
    # Example usage with a sample username
    username = "Agi"
    data = fetch_hiscores(username)
    if data:
        print(f"Hiscores for {username}:")
        for entry in data:
            print(entry)
    else:
        print("Failed to fetch hiscores.")
