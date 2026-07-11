def format_confidence(confidence):
    """Turns 0.87 into '87%'"""
    return f"{confidence * 100:.0f}%"

def capitalize_label(label):
    """Turns 'cell phone' into 'Cell Phone'"""
    if label is None:
        return "Unknown"
    return label.title()