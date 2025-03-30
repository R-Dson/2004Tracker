import requests
from utils.exp import get_xp_for_level

# API type IDs map to these skills
VALID_SKILLS = [
    "Overall",       # 0
    "Attack",        # 1
    "Defence",       # 2
    "Strength",      # 3
    "Hitpoints",     # 4
    "Ranged",        # 5
    "Prayer",        # 6
    "Magic",         # 7
    "Cooking",       # 8
    "Woodcutting",   # 9
    "Fletching",     # 10
    "Fishing",       # 11
    "Firemaking",    # 12
    "Crafting",      # 13
    "Smithing",      # 14
    "Mining",        # 15
    "Herblore",      # 16
    "Agility",       # 17
    "Thieving",      # 18
    "Runecrafting"   # 21
] 

def fetch_hiscores(username):
    """
    Fetch the hiscores data for the given username using the API.
    
    Each dictionary contains:
        - skill: Skill name
        - rank: Rank (integer)
        - level: Level (integer)
        - xp: XP (integer, divided by 10 and truncated)
        
    Returns None if an error occurs.
    """
    url = f"https://2004.lostcity.rs/api/hiscores/player/{username}"
    print(f"Fetching URL: {url}")
    try:
        response = requests.get(url)
        print(f"Response code: {response.status_code}")
        response.raise_for_status()
        api_data = response.json()
    except Exception as e:
        print(f"Error fetching hiscores API for {username}: {e}")
        return None

    # Create a mapping of type to entry for easier lookup
    skill_map = {entry["type"]: entry for entry in api_data}

    hiscores = []
    
    # Map API types to skills, with special handling for Runecrafting
    type_mapping = {
        i: skill for i, skill in enumerate(VALID_SKILLS[:-1])  # All skills except Runecrafting
    }
    # Add Runecrafting with type 21
    type_mapping[21] = "Runecrafting"
    
    # Process skills in VALID_SKILLS order
    for skill_name in VALID_SKILLS:
        # Find the type_id for this skill
        type_id = next(tid for tid, name in type_mapping.items() if name == skill_name)
        
        entry = skill_map.get(type_id, None)
        if entry:
            hiscores.append({
                "skill": skill_name,
                "rank": entry["rank"],
                "level": entry["level"],
                "xp": entry["value"] // 10  # API stores XP * 10, divide and truncate
            })
        else:
            hiscores.append({
                "skill": skill_name,
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
        last_interval_rate = 1 # Store the rate of the last interval processed, default 1 to avoid division by zero

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
            last_interval_rate = rate # Update with the rate of the current interval
 
        # Add EHP for XP gained beyond 99
        if remaining_xp > 0 and last_interval_rate > 0:
            skill_ehp += remaining_xp / last_interval_rate

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
