# requirement pycurl
# pip install pycurl

#!/usr/bin/env python3


import pycurl
import json
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

REST_API_BASEURL = '://localhost:10000/'

class ipm_rest_api():
    """ Rest API  """      

    def send_request(self,server_connection, method, url, payload=None):
        
        #setup
        self.headers = ["Content-Type:application/json"]
        
        buffer = BytesIO()
        conn = pycurl.Curl()

        if method == 'get':
            conn.setopt(pycurl.HTTPGET, 1)
            if payload != None:
                conn.setopt(pycurl.POSTFIELDS, payload)
        elif method == 'post':
            conn.setopt(pycurl.POST, 1)
            if payload == None:
                conn.setopt(pycurl.POSTFIELDS, '')
            else:
                conn.setopt(pycurl.POSTFIELDS, payload)
        
        conn.setopt(pycurl.URL, url)

        if(server_connection == "https"):
            conn.setopt(pycurl.HTTP_VERSION, pycurl.CURL_HTTP_VERSION_2_0)
        else:
            # prior knowledge needed for http
            conn.setopt(pycurl.HTTP_VERSION, pycurl.CURL_HTTP_VERSION_2_PRIOR_KNOWLEDGE)

        conn.setopt(pycurl.NOSIGNAL, 1)
        conn.setopt(pycurl.CONNECTTIMEOUT, 30)
        conn.setopt(pycurl.TIMEOUT, 80)
        conn.setopt(pycurl.HTTPHEADER, self.headers)

        #cacert_path = "./home/sdp/IPM/ipm/server.key"
        if(server_connection == "https"):
            conn.setopt(pycurl.SSL_VERIFYPEER, 1)
            conn.setopt(pycurl.SSL_VERIFYHOST, 2)
            cacert_path = input("Enter cacert_path: ")
            conn.setopt(pycurl.CAINFO, cacert_path)

        conn.setopt(pycurl.WRITEDATA, buffer)
        try:
            conn.perform()
        except Exception as e:
            #print("\n")
            print(e)

        return conn, buffer.getvalue().decode('utf-8')

    def convertResponseToJsonDict(self, response):
        #json.dumps() will convert the response to string type and eval() will reverse it.
        #replace function to remove unicode characters from stri
        json_str = eval(json.dumps(response).replace(r'\u0000', ''))
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except json.decoder.JSONDecodeError:
            print("Empty response")  

    def get_response(self, conn, resp):
        print("Response:",conn.getinfo(pycurl.RESPONSE_CODE))
        json_dict = self.convertResponseToJsonDict(resp)
        print(json_dict)
        return
    
    def get_global_status(self,server_connection):
        ''' Use this API request to return the status of the power control loop main application (running or not running). The server will respond with the status of the application i.e. running | not running '''
        url = server_connection + REST_API_BASEURL +'global/status'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()

    def get_global_configuration(self,server_connection):
        ''' Use this API request to return the global configuration of the UPM. The server will respond with the current global application set (which may be a default set)'''
        url = server_connection + REST_API_BASEURL +'global/configuration'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()
        
    def get_instance_status(self,server_connection):
        ''' Use this API request to return a list of the id's of all the configured instances. Includes a number at the start indicating the number of instances to follow. The server will respond with a list of the id’s of all the configured instances '''
        url = server_connection + REST_API_BASEURL +'instance/status'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()
        
    def get_instance_configuration(self,server_connection,payload):
        ''' Given an instance id, it returns the configuration of that instance (cores, enabled status, frequencies, etc). Only one instance can be queried at a time'''     
        # example of id -- dpdkapp1 or vppapp1
        url = server_connection + REST_API_BASEURL +'instance/configuration'
        conn, resp = self.send_request(server_connection,'get',url,payload)
        self.get_response(conn, resp)
        conn.close()

    def set_global_startpowercontrol(self,server_connection):
        ''' Use this API call to enable monitoring on all configured instances '''
        url = server_connection + REST_API_BASEURL +'global/startpowercontrol'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def set_global_stoppowercontrol(self,server_connection):
        ''' Use this API call to disable monitoring on all configured instances '''
        url = server_connection + REST_API_BASEURL +'global/stoppowercontrol'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def set_global_configuration(self,server_connection,payload):
        ''' Use this API call to modify global parameters that are part of the global config (config.json).'''
        url = server_connection + REST_API_BASEURL +'global/configuration'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()

    def set_instance_create(self,server_connection,payload):
        ''' Use this API call to create an instance of an application to be monitored by IPM. Receives an input list similar to -> list : [ { "id" : "dpdkapp3","telemetry path": "/var/run/dpdk/test112/dpdk_telemetry.v2", "enabled": true, "cores": "15-16","upper": "default", "lower": "default", "mode": "dpdk", "capacity factor": "30", "maxreconnects": 20, "telemetry timeout": 250, "telemetry timeout rocket": true } ] '''
        url = server_connection + REST_API_BASEURL +'instance/create'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()

    def set_instance_delete(self,server_connection,payload):
        ''' Use this API to delete an instance of an application that is monitored by IPM.  Receives an input list similar to -> list : [ { "id" : "dpdkapp3" } ] '''
        url = server_connection + REST_API_BASEURL +'instance/delete'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_instance_update(self,server_connection,payload):
        ''' Use this API call to update the parameters of a running instance. Receives an input list similar to -> list :  [ { "id" : "dpdkapp3", "enabled" : true} ] '''
        url = server_connection + REST_API_BASEURL +'instance/update'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_instance_disable(self,server_connection,payload):
        ''' Use this API call to disable a running instance in IPM.  Receives an input list similar to -> list :  [ {"id": "dpdkapp3"} ] '''
        url = server_connection + REST_API_BASEURL +'instance/disable'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()

    def set_instance_enable(self,server_connection,payload):
        ''' Use this API call to enable an instance in IPM.  Receives an input list similar to -> list :  [ {"id": "dpdkapp3"} ] '''
        url = server_connection + REST_API_BASEURL +'instance/enable'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def get_provision_cstates_configuration(self,server_connection):
        ''' Use this API call to display the C State configuration on the system '''
        url = server_connection + REST_API_BASEURL +'provision/cstates/configuration'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()

    def get_provision_cstates_info(self,server_connection):
        ''' Use this API call to get the current and available C state settings on the system '''
        url = server_connection + REST_API_BASEURL +'provision/cstates/info'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()

    def get_provision_pstates_configuration(self,server_connection):
        ''' Use this API call to display the P state configuration on the system '''
        url = server_connection + REST_API_BASEURL +'provision/pstates/configuration'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()

    def set_provision_cstates_create(self,server_connection,payload):
        ''' Use this API call to set a C state for a range of cores on the system. Receives two inputs governor and c1 --> governor example - "ladder", c1 example { "enable cores" : "10-12", "disable cores" : "13, 14"}} '''
        url = server_connection + REST_API_BASEURL +'provision/cstates/create'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_provision_cstates_update(self,server_connection,payload):
        ''' Use this API call to update the C state setting for a range of cores. Receives two inputs governor and c6 --> governor example - "menu", c6 example {"enable cores" : "10-12", "disable cores" : "13, 14"}}'''
        url = server_connection + REST_API_BASEURL +'provision/cstates/update'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()

    def set_provision_cstates_delete(self,server_connection):
        ''' Use this API call to delete the C state configuration. This will revert changed values to their prior state. '''
        url = server_connection + REST_API_BASEURL +'provision/cstates/delete'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def set_provision_pstates_create(self,server_connection,payload):
        ''' Use this API call to set a P state for a range of core. Receives inputs "global" example {"max perf pct":95, "no turbo": 0} and "group1" example {"cores": "15-20", "lower": "1000", "upper": "1800", "governor":"performance"} '''
        url = server_connection + REST_API_BASEURL +'provision/pstates/create'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_provision_pstates_update(self,server_connection,payload):
        ''' Use this API call to update the P state for a range of cores. Receives input "group2" example {"cores": "43","energy performance preference": "balance_power"}}'''
        url = server_connection + REST_API_BASEURL +'provision/pstates/update'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_provision_pstates_delete(self,server_connection):
        ''' Use this API call to delete the P state configuration. This will revert changed values to their prior state. '''
        url = server_connection + REST_API_BASEURL +'provision/pstates/delete'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()
        
    def get_ufs_status(self,server_connection):
        ''' Use this API call to get the uncore frequency scaling current status (enabled or disabled) '''
        url = server_connection + REST_API_BASEURL +'ufs/status'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()
        
    def get_ufs_configuration(self,server_connection):
        ''' Use this API call to get the current configuration of uncore frequency scaling '''
        url = server_connection + REST_API_BASEURL +'ufs/configuration'
        conn, resp = self.send_request(server_connection,'get',url)
        self.get_response(conn, resp)
        conn.close()

    def set_ufs_create(self,server_connection,payload):
        ''' Use this API call to create a UFS configuration and apply it to specific cores. Receives inputs "enabled" example true and "groups" example [{ "group id":"group 1", "telemetry cores": "5-7", "smoothing samples": 5}]} '''
        url = server_connection + REST_API_BASEURL +'ufs/create'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_ufs_delete(self,server_connection):
        ''' Use this API call to delete any existing uncore frequency scaling configuration '''
        url = server_connection + REST_API_BASEURL +'ufs/delete'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def set_ufs_update(self,server_connection,payload):
        ''' Use this API call to update an existing uncore frequency scaling configuration. Receives inputs "enabled" example true and "groups" example [{ "group id":"group 1", "telemetry cores": "5-7", "smoothing samples": 10}] . Note: When calling the /ufs/update API the entire updated configuration (both changed and unchanged values) for uncore frequency scaling must be applied. Manipulation of individual parameters of the configuration is not supported through the API'''
        url = server_connection + REST_API_BASEURL +'ufs/update'
        conn, resp = self.send_request(server_connection,'post',url,payload)
        self.get_response(conn, resp)
        conn.close()
        
    def set_ufs_enable(self,server_connection):
        ''' Use this API call to enable uncore frequency scaling '''
        url = server_connection + REST_API_BASEURL +'ufs/enable'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def set_ufs_disable(self,server_connection):
        ''' Use this API call to disable uncore frequency scaling '''
        url = server_connection + REST_API_BASEURL +'ufs/disable'
        conn, resp = self.send_request(server_connection,'post',url)
        self.get_response(conn, resp)
        conn.close()

    def handle_rest_api():
        
        while True:

            print("Choose REST API ")
            print("1. GET global/status ")
            print("2. GET global/configuration ")
            print("3. GET instance/status ")
            print("4. GET instance/configuration ") 
            print("5. POST global/startpowercontrol ")
            print("6. POST global/stoppowercontrol ")
            print("7. POST global/configuration ")
            print("8. POST instance/create ")
            print("9. POST instance/delete ")
            print("10. POST instance/update ")
            print("11. POST instance/disable ")
            print("12. POST instance/enable ")
            print("13. GET provision/cstates/configuration ")
            print("14. GET provision/cstates/info ")
            print("15. GET provision/pstates/configuration ")
            print("16. POST provision/cstates/create ")
            print("17. POST provision/cstates/update ")
            print("18. POST provision/cstates/delete ")
            print("19. POST provision/pstates/create ")
            print("20. POST provision/pstates/update ")
            print("21. POST provision/pstates/delete ")
            print("22. GET ufs/status ")
            print("23. GET ufs/configuration ")
            print("24. POST ufs/create")
            print("25. POST ufs/delete")
            print("26. POST ufs/update")
            print("27. POST ufs/enable ")
            print("28. POST ufs/disable ")
            print("29. Go Back")

            choice = input("Enter your choice (or '29' to go back): ")
            
            if choice == '29':
                break  # Exit the function to go back to the previous menu

            print("Choose server_connection ")
            print("1. http ")
            print("2. https ") # https yet to be tested just added 
            

            server_connection_choice = input("Choose server connection (1. http 2. https): ")
            if server_connection_choice == '1':
                server_connection = 'http'
            elif server_connection_choice == '2':
                server_connection = 'https'
            else:
                print("Invalid server connection choice. Please enter 1 for 'http' or 2 for 'https'.")

            ipm_rest_api_obj = ipm_rest_api()
            
            if choice == '1':
                try:
                    response = ipm_rest_api_obj.get_global_status(server_connection)
                    print("Global Status:", response)
                except Exception as e:
                    print(f"An error occurred while fetching global status: {e}")
                
            elif choice == '2':
                try:
                    response = ipm_rest_api_obj.get_global_configuration(server_connection)
                    print("Global Configuration:", response)
                except Exception as e:
                    print(f"An error occurred while fetching global configuration: {e}")
                
            elif choice == '3':
                try:
                    response = ipm_rest_api_obj.get_instance_status(server_connection)
                    print("Instance Status:", response)
                except Exception as e:
                    print(f"An error occurred while fetching instance status: {e}")
                
            elif choice == '4':
                # example of id -- dpdkapp1 or vppapp1
                data = {
                    "id" : "dpdkapp1"
                    }
                try:
                    payload = json.dumps(data)
                    response = ipm_rest_api_obj.get_instance_configuration(server_connection,payload)
                    print("Instance Configuration:", response)
                except Exception as e:
                    print(f"An error occurred while fetching instance configuration: {e}")
                    
            elif choice == '5':
                try:
                    response = ipm_rest_api_obj.set_global_startpowercontrol(server_connection)
                    print("Set Global Configuration:", response)
                except Exception as e:
                    print(f"An error occurred while setting global configuration: {e}")
                    
            elif choice == '6':
                print(ipm_rest_api_obj.set_global_stoppowercontrol(server_connection))
            elif choice == '7':
                data = {
                    "loop time": 4000,
                    "log level": "DEBUG"
                    }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_global_configuration(server_connection,payload))
            elif choice == '8':
                data = {
                        "list" : [
                                    { 
                                        "id" : "dpdkapp3",
                                        "telemetry path": "/var/run/dpdk/test112/dpdk_telemetry.v2", 
                                        "enabled": True, 
                                        "cores": "15-16",
                                        "upper": "default", 
                                        "lower": "default", 
                                        "mode": "dpdk", 
                                        "capacity factor": "30", 
                                        "max reconnects": 20, 
                                        "telemetry timeout": 250, 
                                        "telemetry timeout rocket": True 
                                    }
                                ]
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_instance_create(server_connection,payload))
            elif choice == '9':
                data = {
                        "list" : [ 
                                    { 
                                        "id" : "dpdkapp3"
                                    } 
                                ]
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_instance_delete(server_connection,payload))
            elif choice == '10':
                data = { 
                            "list" : [ 
                                        {
                                            "id" : "dpdkapp3", 
                                            "enabled" : True
                                        } 
                                    ] 
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_instance_update(server_connection,payload))
            elif choice == '11':
                data = {
                            "list" : [
                                        {   
                                            "id": "dpdkapp3"
                                        } 
                                    ] 
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_instance_disable(server_connection,payload))
            elif choice == '12':
                data = {
                            "list" : [
                                        {   
                                            "id": "dpdkapp3"
                                        } 
                                    ] 
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_instance_enable(server_connection,payload))
            elif choice == '13':
                print(ipm_rest_api_obj.get_provision_cstates_configuration(server_connection))
            elif choice == '14':
                print(ipm_rest_api_obj.get_provision_cstates_info(server_connection))
            elif choice == '15':
                print(ipm_rest_api_obj.get_provision_pstates_configuration(server_connection))
            elif choice == '16':
                data = { 
                            "governor" : "ladder", 
                            "c1": { 
                                    "enable cores" : "10-12", 
                                    "disable cores" : "13, 14"
                                    }
                        }
                        
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_provision_cstates_create(server_connection,payload))
            elif choice == '17':
                data = { 
                            "governor" : "menu", 
                            "c6": { 
                                        "enable cores" : "10-12", 
                                        "disable cores" : "13, 14"
                                    }
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_provision_cstates_update(server_connection,payload))
            elif choice == '18':
                print(ipm_rest_api_obj.set_provision_cstates_delete(server_connection))
            elif choice == '19':
                data = {
                            "global" : {
                                            "max perf pct":95,
                                            "no turbo": 0
                                        }, 
                            "group1": {
                                            "cores": "15-20", 
                                            "lower": "1000", 
                                            "upper": "1800", 
                                            "governor": "performance"
                                        }
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_provision_pstates_create(server_connection,payload))
            elif choice == '20':
                data = {
                            "group2": {
                                            "cores": "43",
                                            "energy performance preference": "balance_power"
                                        }
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_provision_pstates_update(server_connection,payload))
            elif choice == '21':
                print(ipm_rest_api_obj.set_provision_pstates_delete(server_connection))
            elif choice == '22':
                print(ipm_rest_api_obj.get_ufs_status(server_connection))
            elif choice == '23':
                print(ipm_rest_api_obj.get_ufs_configuration(server_connection))
            elif choice == '24':
                data = {
                            "enabled": True, 
                            "groups": [
                                        { 
                                            "group id":"group 1", 
                                            "telemetry cores": "5-7", 
                                            "smoothing samples": 5
                                        }
                                    ]
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_ufs_create(server_connection,payload))
            elif choice == '25':
                print(ipm_rest_api_obj.set_ufs_delete(server_connection))
            elif choice == '26':
                data = {
                            "enabled": True, 
                            "groups": [
                                            { 
                                                "group id":"group 1", 
                                                "telemetry cores": "5-7", 
                                                "smoothing samples": 10
                                            }
                                        ]
                        }
                payload = json.dumps(data)
                print(ipm_rest_api_obj.set_ufs_update(server_connection,payload))
            elif choice == '27':
                print(ipm_rest_api_obj.set_ufs_enable(server_connection))
            elif choice == '28':
                print(ipm_rest_api_obj.set_ufs_disable(server_connection))
            else:
                print("Invalid choice!")