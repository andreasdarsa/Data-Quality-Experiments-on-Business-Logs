import random
import pandas as pd
from datetime import datetime, timedelta

ACTIVITIES = [
    "Receive Order",
    "Validate Order",
    "Approve Order",
    "Execute Order",
    "Close Order"
]

DURATIONS = {
    "Receive Order": (1, 3),
    "Validate Order": (2, 5),
    "Approve Order": (1, 4),
    "Execute Order": (5, 10),
    "Close Order": (1, 2)
}

def generate_case(case_id, start_time):
    events = []
    current_time = start_time

    def add_event(activity):
        nonlocal current_time
        duration = random.randint(*DURATIONS[activity])
        start = current_time
        end = start + timedelta(minutes=duration)

        events.append({
            "case_id": case_id,
            "activity": activity,
            "start_timestamp": start,
            "complete_timestamp": end
        })

        current_time = end

    # Receive
    add_event("Receive Order")

    # Validate
    add_event("Validate Order")

    # Rejection path
    if random.random() < 0.2:
        add_event("Close Order")
        return events

    # Approval
    add_event("Approve Order")

    # Possible loop in execution
    add_event("Execute Order")
    if random.random() < 0.1:
        add_event("Execute Order")

    # Close
    add_event("Close Order")

    return events

def generate_log(num_cases):
    log = []
    base_time = datetime.now() - timedelta(days=30)

    for case_id in range(1, num_cases + 1):
        start_time = base_time + timedelta(minutes=random.randint(0, 43200))  # Random start within last 30 days
        case_events = generate_case(case_id, start_time)
        log.extend(case_events)

    log_df = pd.DataFrame(log)
    log_df.to_csv("data/synthetic_event_log.csv", index=False)
    return log_df

logs = generate_log(1000)
print(logs.head())