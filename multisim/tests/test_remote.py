import paramiko
import json

def execute_remote_script_via_ssh(hostname, username, password, script_path):
    # Establish SSH connection
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)

    # Execute the Python script remotely
    stdin, stdout, stderr = client.exec_command('python {}'.format(script_path))
    
    # Read the output of the script
    output = stdout.read().decode('utf-8')

    # Close the SSH connection
    client.close()

    return output

def main():
    # SSH connection details
    hostname = 'your_remote_host'
    username = 'your_username'
    password = 'your_password'

    # Path to the Python script on the remote host
    script_path = '/path/to/your/script.py'

    # Execute remote script and get the JSON output
    output = execute_remote_script_via_ssh(hostname, username, password, script_path)

    # Load the returned JSON data
    try:
        json_data = json.loads(output)
        print("Returned JSON data:")
        print(json.dumps(json_data, indent=4))
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

if __name__ == "__main__":
    main()