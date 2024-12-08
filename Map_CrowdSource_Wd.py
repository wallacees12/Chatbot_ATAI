import csv
import requests
import json

# Step 1: Read the TSV and extract unique wd and wdt IDs
with open('crowd_data.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')

    unique_wd = set()
    unique_wdt = set()

    for row in reader:
        for col in ['Input1ID', 'Input2ID', 'Input3ID']:
            val = row.get(col, '')
            if val.startswith("wd:"):
                # Extract the QID part (e.g. Q11621)
                qid = val.split(':')[1]
                unique_wd.add(qid)
            elif val.startswith("wdt:"):
                # Extract the PID part (e.g. P2142)
                pid = val.split(':')[1]
                unique_wdt.add(pid)


def get_labels_for_entities(entities):
    """
    Given a list of QIDs or PIDs (e.g., ["Q11621", "P2142"]), 
    fetch their English labels from Wikidata.
    """
    if not entities:
        return {}
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(entities),
        "props": "labels",
        "languages": "en",
        "format": "json"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    results = {}
    for eid, info in data.get('entities', {}).items():
        label = info.get('labels', {}).get('en', {}).get('value', eid)
        results[eid] = label
    return results

# Convert sets to lists
qids = list(unique_wd)
pids = list(unique_wdt)

# Step 2: Fetch labels for QIDs and PIDs
entity_labels = get_labels_for_entities(qids)
property_labels = get_labels_for_entities(pids)

# Construct a JSON object
output = {
    "entity_labels": entity_labels,
    "property_labels": property_labels
}

# Save the JSON object to a file
with open('labels.json', 'w', encoding='utf-8') as json_file:
    json.dump(output, json_file, indent=2, ensure_ascii=False)