import grpc
import json
import sys
import csv
from datetime import datetime, timedelta
import tac_pb2
import tac_service_pb2_grpc

def to_iso8601(dt):
    """Converts a datetime object to an ISO 8601 formatted string."""
    return dt.isoformat() + 'Z'

def preprocess_data(data_str):
    """Preprocesses the data string to ensure it is valid JSON."""
    try:
        # Attempt to load the string directly as JSON
        return json.loads(data_str)
    except json.JSONDecodeError:
        # If JSON decoding fails, attempt to evaluate it as a Python dictionary
        try:
            # This is not secure and not recommended for production code
            # Consider using ast.literal_eval() for a safer alternative
            data_dict = eval(data_str)
            return data_dict
        except Exception as e:
            print(f"Failed to preprocess data: {e}")
            return None

def save_to_file(data, filename, format):
    """Saves data to a file in the specified format."""
    with open(filename, 'w', newline='') as file:
        if format == 'json':
            json.dump(data, file, indent=4)
        elif format == 'csv':
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Instance', 'Time', 'Value'])
            for result in data.get('data', {}).get('result', []):
                metric_name = result['metric'].get('__name__', 'unknown metric')
                for timestamp, value in result.get('values', []):
                    readable_time = datetime.utcfromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S UTC')
                    writer.writerow([metric_name, result['metric'].get('instance', ''), readable_time, value])

def display_output(data, format):
    """Displays data in the specified format."""
    if format == 'json':
        print(json.dumps(data, indent=4))
    elif format == 'csv':
        writer = csv.writer(sys.stdout)  # Use sys.stdout as the file-like object
        writer.writerow(['Metric', 'Instance', 'Time', 'Value'])
        for result in data.get('data', {}).get('result', []):
            metric_name = result['metric'].get('__name__', 'unknown metric')
            for timestamp, value in result.get('values', []):
                readable_time = datetime.utcfromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S UTC')
                instance = result['metric'].get('instance', '')
                writer.writerow([metric_name, instance, readable_time, value])
def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = tac_service_pb2_grpc.TacServiceStub(channel)

    # User input for metric name with 'up' as default
    metric_name = input("Enter metric name (default 'up'): ") or 'up'

    now = datetime.utcnow()
    start_time = now - timedelta(hours=1)
    end_time = now

    metrics_query = tac_pb2.MetricsQuery(
        metric=metric_name,
        time_range=tac_pb2.TimeRange(
            start=to_iso8601(start_time),
            end=to_iso8601(end_time)
        ),
        aggregation='sum',
        interval='5m'
    )

    response = stub.CollectMetrics(metrics_query)

    # Preprocess the data before attempting to parse as JSON
    data = preprocess_data(response.data)

    if data:
    	# User input for output format
    	format = input("Select output format - 1 for JSON, 2 for CSV: ")
    	format = 'json' if format == '1' else 'csv'

    	# User input for output destination
    	output_destination = input("Select output destination - 1 for stdout, 2 for file: ")
    
    	if output_destination == '1':
        	display_output(data, format)
    	elif output_destination == '2':
        	filename = input("Enter filename: ")
        	save_to_file(data, filename, format)
        	print(f"Data saved to {filename}")
    else:
        print("Failed to parse the response data.")

if __name__ == '__main__':
    run()
