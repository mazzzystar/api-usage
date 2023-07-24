import os
from flask import Flask, render_template, request
import requests
import datetime
import json
from collections import defaultdict

app = Flask(__name__)

api_key = os.environ['API_KEY']
headers = {'Authorization': f'Bearer {api_key}'}
url = 'https://api.openai.com/v1/usage'

def get_usage(date):
    params = {'date': date}
    response = requests.get(url, headers=headers, params=params)
    usage_data = response.json()['data']
    return usage_data


@app.route('/', methods=['GET'])
def index():
    granularity = request.args.get('granularity', '60')  # Get granularity from query parameters, default is '5'
    date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    usage_data = get_usage(date)

    # Group usage data by timestamp and model
    usage_by_timestamp = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data in usage_data:
        timestamp = datetime.datetime.fromtimestamp(data['aggregation_timestamp'])
        timestamp = (timestamp - datetime.timedelta(minutes=timestamp.minute % int(granularity),
                                                    seconds=timestamp.second)).strftime('%m%d %H:%M')
        model = data['snapshot_id']
        context_tokens = data['n_context_tokens_total']
        generated_tokens = data['n_generated_tokens_total']

        # Calculate cost based on model
        if model.startswith('gpt-3.5-turbo'):
            context_cost = context_tokens / 1000 * 0.0015
            generated_cost = generated_tokens / 1000 * 0.002
        elif model == 'text-embedding-ada-002-v2':
            context_cost = context_tokens / 1000 * 0.0001
            generated_cost = 0  # No cost for generated tokens
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
        context_costs = [usage_by_timestamp[timestamp][model]['context_cost'] for timestamp in timestamps]
        generated_costs = [usage_by_timestamp[timestamp][model]['generated_cost'] for timestamp in timestamps]
        if model != 'text-embedding-ada-002-v2':
            datasets.append({'label': f'{model} (context)', 'data': context_costs, 'stack': 'Stack 0'})
            datasets.append({'label': f'{model} (generated)', 'data': generated_costs, 'stack': 'Stack 0'})
        else:
            total_costs = [context_costs[i] + generated_costs[i] for i in range(len(context_costs))]
            datasets.append({'label': f'{model} (total)', 'data': total_costs, 'stack': 'Stack 0'})

    return render_template('index.html', timestamps=json.dumps(timestamps), datasets=json.dumps(datasets), granularity=granularity)


if __name__ == '__main__':
    app.run(debug=True)
