# tac.py

from abc import ABC, abstractmethod
from concurrent import futures
import grpc
import requests
import configparser

# Import the generated Protobuf code
import generated.tac_pb2 as tac_pb2
import generated.tac_pb2_grpc as tac_service_pb2_grpc

# Abstract Base Classes for Telemetry Operations

class MetricsCollector(ABC):
    @abstractmethod
    def query_metrics(self, query):
        pass

class LogsCollector(ABC):
    @abstractmethod
    def query_logs(self, query):
        pass

class DashboardManager(ABC):
    @abstractmethod
    def create_dashboard(self, config):
        pass

    @abstractmethod
    def update_dashboard(self, config):
        pass

# Concrete Implementations for Each Telemetry System

class PrometheusMetricsCollector(MetricsCollector):
    def query_metrics(self, query):
        # Prometheus-specific metric querying logic
        response = requests.get('http://prometheus-url/api/v1/query', params={'query': query})
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Failed to query Prometheus metrics")

class LokiLogsCollector(LogsCollector):
    def query_logs(self, query):
        # Loki-specific logs querying logic
        response = requests.get('http://loki-url/api/v1/query', params={'query': query})
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Failed to query Loki logs")
        return "Loki logs"

class GrafanaDashboardManager(DashboardManager):
    def create_dashboard(self, config):
        # Grafana-specific dashboard creation logic
        response = requests.post('http://grafana-url/api/dashboards/db', json=config)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Failed to create Grafana dashboard")
        return f"Dashboard {config.name} created in Grafana"

    def update_dashboard(self, config):
        # Grafana-specific dashboard update logic
        response = requests.put('http://grafana-url/api/dashboards/db/{dashboard_id}', json=config)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Failed to update Grafana dashboard")
        return f"Dashboard {config.name} updated in Grafana"

# Factory Classes for Telemetry System Instantiation

class MetricsCollectorFactory:
    @staticmethod
    def get_collector(source):
        if source == "Prometheus":
            return PrometheusMetricsCollector()
        # Add more conditions for other telemetry systems as needed
        raise ValueError("Unsupported metrics source")

class LogsCollectorFactory:
    @staticmethod
    def get_collector(source):
        if source == "Loki":
            return LokiLogsCollector()
        # Add more conditions for other telemetry systems as needed
        raise ValueError("Unsupported logs source")

class DashboardManagerFactory:
    @staticmethod
    def get_manager():
        # Currently only Grafana is supported, but this can be extended
        return GrafanaDashboardManager()

# TacService gRPC Service Implementation

class TacService(tac_service_pb2_grpc.TacServiceServicer):
    def __init__(self, metrics_collector, logs_collector, dashboard_manager):
        self.metrics_collector = metrics_collector
        self.logs_collector = logs_collector
        self.dashboard_manager = dashboard_manager

    def CollectMetrics(self, request, context):
        try:
            result = self.metrics_collector.query_metrics(request.query)
            return tac_pb2.QueryResponse(status="success", data=str(result))  # Ensure data is serialized properly
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error collecting metrics: {str(e)}')
            return tac_pb2.QueryResponse(status="error", error=str(e))

    def QueryMetrics(self, request, context):
        # Implement similarly to CollectMetrics, or merge functionality if they're the same
        pass

    def CollectLogs(self, request, context):
        try:
            result = self.logs_collector.query_logs(request.query)
            return tac_pb2.QueryResponse(status="success", data=str(result))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error collecting logs: {str(e)}')
            return tac_pb2.QueryResponse(status="error", error=str(e))

    def QueryLogs(self, request, context):
        # Implement similarly to CollectLogs, or merge functionality if they're the same
        pass

    def CreateDashboard(self, request, context):
        try:
            result = self.dashboard_manager.create_dashboard(request)
            return tac_pb2.QueryResponse(status="success", data=str(result))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error creating dashboard: {str(e)}')
            return tac_pb2.QueryResponse(status="error", error=str(e))
    
    def UpdateDashboard(self, request, context):
        try:
            result = self.dashboard_manager.update_dashboard(request)
            return tac_pb2.QueryResponse(status="success", data=str(result))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error updating dashboard: {str(e)}')
            return tac_pb2.QueryResponse(status="error", error=str(e))
                
config = configparser.ConfigParser()

def get_telemetry_systems():
    metrics_system = config.get('TelemetrySystems', 'Metrics')
    logs_system = config.get('TelemetrySystems', 'Logs')
    return metrics_system, logs_system

metrics_system, logs_system = get_telemetry_systems()
metrics_collector = MetricsCollectorFactory.get_collector(metrics_system)
logs_collector = LogsCollectorFactory.get_collector(logs_system)
dashboard_manager = DashboardManagerFactory.get_manager()  # Assuming Grafana for now

service = TacService(metrics_collector, logs_collector, dashboard_manager)

# Function to Start the gRPC Server

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tac_pb2_grpc.add_TacServiceServicer_to_server(TacService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on '[::]:50051'")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
