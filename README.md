# MasterProject2018
## Setup Instructions

### Package manager - pip

To install pip on Mac while using the in-built python install:

`$ sudo easy_install pip` 

### Python Virtual Environment - virtualenv

To isolate this project from the system level python install and associated packages you can use 'virtualenv'. In short:

1 - Install virtualenv with pip

`$ sudo pip install virtualenv`

>The above steps might not work and were failing to install/upgrade to pip 9.0.3 and virtualenv 1.5.2, which in turn, was causing dependencies not to install (not found) when running pip -r ...

>Installing pip using the following method, did install 9.0.3 which then fetched virtualenv 1.5.2. 

>MacBook$ sudo -H python get-pip.py
Collecting pip
  Downloading pip-9.0.3-py2.py3-none-any.whl (1.4MB)
    100% |████████████████████████████████| 1.4MB 383kB/s 
Collecting wheel
  Downloading wheel-0.31.0-py2.py3-none-any.whl (41kB)
    100% |████████████████████████████████| 51kB 484kB/s 
Installing collected packages: pip, wheel
  Found existing installation: pip 9.0.1
    Uninstalling pip-9.0.1:
      Successfully uninstalled pip-9.0.1
Successfully installed pip-9.0.3 wheel-0.31.0
>Macbook$ sudo -H pip install virtualenv
Collecting virtualenv
  Downloading virtualenv-15.2.0-py2.py3-none-any.whl (2.6MB)
    100% |████████████████████████████████| 2.6MB 212kB/s 
Installing collected packages: virtualenv
Successfully installed virtualenv-15.2.0"

2 - Create a virtual env in the project folder ( places everything in new 'env/' folder):

`$ virtualenv env`

3 - 'Activate' this environment

`$ source env/bin/activate`

4 - Now using env contained versions of pip and python. You can install the exact dependencies (currently there are just some initial packages at the moment, may need more dependencies added for those packages and will need to be extended based on actual packages used etc,

`$ pip install -r "requirements.txt"`

5 - Ready to go. Use script comfortably with exact versions for all dependencies.

6 - When done using. Return terminal to normal environment

`$ deactivate`

note: make sure the path to the project folder hasn't got spaces, this seems to cause issues with pip inside a virtualenv atm. 
