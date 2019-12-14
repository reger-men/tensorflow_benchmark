from paramiko import SSHClient, AutoAddPolicy
from paramiko.auth_handler import AuthenticationException
from paramiko.ssh_exception import NoValidConnectionsError

class Config(object):
    """Worker access data"""
    
    def __init__(self, ip, user, pwd, port=22):
        self.ip = ip
        self.port = port
        self.user = user
        self.pwd = pwd

class Worker(object):
    """Worker Object to connect and execute commands from the  'chief' worker"""
    
    def __init__(self, config):
        self.ip = config.ip
        self.port = config.port
        self.user = config.user
        self.pwd = config.pwd
        self.client = None
        self.policy = AutoAddPolicy()

    def connect(self):
        """Connect to the worker"""

        if self.client is None:
            try:
                client = SSHClient()
                client.set_missing_host_key_policy(self.policy)
                client.connect(hostname=self.ip,
                               port=self.port,
                               username=self.user,
                               password=self.pwd)

            except AuthenticationException:
                print("Authentication failed!")
            except NoValidConnectionsError:
                print("Connection failed!")    
            finally:
                client.exec_command("hostnamectl")
                return client
        return self.client

    def exec_cmd(self, cmd, inBackground=False, timeout=None):
        """Execute command and return status and output"""
        """ status 0 means no error"""
        status=0
        stdout='Process run in background'

        self.client = self.connect()
        if inBackground:
            transport = self.client.get_transport()
            channel = transport.open_session()
            channel.setblocking(0)
            channel.exec_command(cmd)

        else:
            stdin, stdout, stderr = self.client.exec_command(cmd)
            status = stdout.channel.recv_exit_status()
            if status != 0:
                stdout = stderr
        return status, stdout

    def disconnect(self):
        self.client.close()

