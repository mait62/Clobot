STORAGE_MAP = {
    "cup": "Kitchen cabinet",
    "bottle": "Kitchen cabinet",
    "book": "Bookshelf",
    "cell phone": "Charging station",
    "laptop": "Study desk",
    "scissors": "Drawer",
    "knife": "Kitchen drawer",
    "remote": "TV stand",
    "keyboard": "Study desk",
    "mouse": "Study desk",
    "backpack": "Closet",
    "umbrella": "Entryway stand",
    "shoe": "Shoe rack",
    "clock": "Wall shelf",
    "charger": "Pen Stand"
}

def get_storage_location(object_name):
    """
    Returns a storage suggestion for a detected object.
    Falls back to a generic message if not in our map.
    """
    if object_name is None:
        return "No object detected"
    
    return STORAGE_MAP.get(object_name.lower(), f"No specific rule for '{object_name}' — store in general storage")