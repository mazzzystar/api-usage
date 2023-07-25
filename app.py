import os
from pytz import timezone, utc
import pytz
from flask import Flask, render_template, request
import requests
import datetime
import json
from collections import defaultdict

app = Flask(__name__)

api_key = os.environ.get('OPENAI_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}
url = 'https://api.openai.com/v1/usage'

model_costs = {
    'gpt-3.5-turbo-0301': {'context': 0.0015, 'generated': 0.002},
    'gpt-3.5-turbo-0613': {'context': 0.0015, 'generated': 0.002},
    'gpt-3.5-turbo-16k': {'context': 0.003, 'generated': 0.004},
    'gpt-3.5-turbo-16k-0613': {'context': 0.003, 'generated': 0.004},
    'gpt-4-0314': {'context': 0.03, 'generated': 0.06},
    'gpt-4-0613': {'context': 0.03, 'generated': 0.06},
    'gpt-4-32k': {'context': 0.06, 'generated': 0.12},
    'gpt-4-32k-0314': {'context': 0.06, 'generated': 0.12},
    'gpt-4-32k-0613': {'context': 0.06, 'generated': 0.12},
    'text-embedding-ada-002-v2': {'context': 0.0001, 'generated': 0},  # No cost for generated tokens
    'whisper-1': {'context': 0.006 / 60, 'generated': 0}  # Cost is per second, so convert to minutes
}


def get_usage(api_key, date, use_own_key=False):
    headers = {'Authorization': f'Bearer {api_key}'}
    params = {'date': date}

    if use_own_key:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raises a HTTPError if the response was unsuccessful
            response = response.json()
        except requests.RequestException as e:
            print(f"Request to OpenAI API failed: {e}")
            return None, None
    else:
        """
        To avoid "Request to OpenAI API failed: 429 Client Error: Too Many Requests for url: https://api.openai.com/v1/usage"
        """
        with open('static/data.json', 'r') as f:
            response = json.load(f)
    try:
        usage_data = response['data']
        whisper_data = response['whisper_api_data']
    except KeyError as e:
        print(f"Error processing OpenAI API response: {e}")
        return None, None

    return usage_data, whisper_data


# Convert date to UTC
def to_utc(date):
    local = timezone('America/Los_Angeles')  # Replace with your timezone
    naive = datetime.datetime.strptime(date, '%Y-%m-%d')
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(utc)
    return utc_dt.strftime('%Y-%m-%d')


@app.route('/', methods=['GET', 'POST'])
def index():
    use_own_key = True
    default_date = '2023-07-24'
    date = request.args.get('date')

    # Check if this is a POST request
    if request.method == 'POST':
        # Get the API key from the form data
        api_key = request.form.get('api_key')

        if not api_key:
            # If the API key field was empty, use the default key
            api_key = os.environ.get('OPENAI_API_KEY')
            use_own_key = False
            date = default_date
    else:
        api_key = os.environ.get('OPENAI_API_KEY')  # Default API key for GET requests
        use_own_key = False
        date = default_date

    granularity = request.args.get('granularity', '60')  # Get granularity from query parameters, default is '60'

    if date is None:
        date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    date = to_utc(date)
    usage_data, whisper_data = get_usage(api_key, date, use_own_key)
    if usage_data is None or whisper_data is None:
        # Handle the error (e.g., by showing an error message to the user and returning early)
        return render_template('error.html')

    # Group usage data by timestamp and model
    usage_by_timestamp = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    total_cost = 0

    model_total_costs = defaultdict(float)

    for data in whisper_data:
        model = data['model_id']
        if model in model_costs:
            timestamp = datetime.datetime.fromtimestamp(data['timestamp'])
            timestamp = (timestamp - datetime.timedelta(minutes=timestamp.minute % int(granularity),
                                                        seconds=timestamp.second)).strftime('%m-%d %H:%M')
            num_seconds = data['num_seconds']  # Usage is in seconds
            cost = num_seconds * model_costs[model]['context']  # Calculate cost

            usage_by_timestamp[timestamp][model]['whisper_cost'] += cost
            total_cost += cost  # Add to total cost

            usage_by_timestamp[timestamp][model]['whisper_cost'] += cost
            model_total_costs[model] += cost

    for data in usage_data:
        timestamp = datetime.datetime.fromtimestamp(data['aggregation_timestamp'])
        timestamp = (timestamp - datetime.timedelta(minutes=timestamp.minute % int(granularity),
                                                    seconds=timestamp.second)).strftime('%m-%d %H:%M')
        model = data['snapshot_id']
        context_tokens = data['n_context_tokens_total']
        generated_tokens = data['n_generated_tokens_total']

        # Calculate cost based on model
        if model in model_costs:
            context_cost = context_tokens / 1000 * model_costs[model]['context']
            generated_cost = generated_tokens / 1000 * model_costs[model]['generated']
            total_cost += context_cost + generated_cost
            model_total_costs[model] += context_cost + generated_cost
        else:
            context_cost = generated_cost = 0  # Unknown model

        usage_by_timestamp[timestamp][model]['context_cost'] += context_cost
        usage_by_timestamp[timestamp][model]['generated_cost'] += generated_cost

    # Prepare data for the chart
    # Prepare data for the chart
    timestamps = sorted(usage_by_timestamp.keys())
    model_names = sorted(set(model for usage in usage_by_timestamp.values() for model in usage.keys()))
    datasets = []
    for model in model_names:
        context_costs = [usage_by_timestamp[timestamp][model].get('context_cost', 0) for timestamp in timestamps]
        generated_costs = [usage_by_timestamp[timestamp][model].get('generated_cost', 0) for timestamp in timestamps]
        whisper_costs = [usage_by_timestamp[timestamp][model].get('whisper_cost', 0) for timestamp in timestamps]
        total_costs = [context_costs[i] + generated_costs[i] + whisper_costs[i] for i in range(len(context_costs))]
        if sum(total_costs) > 0:  # Only add model to datasets if its total cost is greater than 0
            if model != 'text-embedding-ada-002-v2' and model != 'whisper-1':
                datasets.append({'label': f'{model} (context)', 'data': context_costs, 'stack': 'Stack 0'})
                datasets.append({'label': f'{model} (generated)', 'data': generated_costs, 'stack': 'Stack 0'})
            else:
                datasets.append({'label': f'{model} (total)', 'data': total_costs, 'stack': 'Stack 0'})

    return render_template('index.html', timestamps=json.dumps(timestamps), datasets=json.dumps(datasets),
                           model_total_costs=json.dumps(dict(model_total_costs)), granularity=granularity, date=date,
                           total_cost=total_cost)


if __name__ == '__main__':
    app.run(debug=True)
