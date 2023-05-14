from datetime import datetime

def time(secs):
    return datetime.timedelta(seconds=round(secs))
