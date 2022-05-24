import os
import subprocess

program_list = ['ex13_1.py', 'ex13_2.py', 'ex13_3.py']

for program in program_list:
    if os.path.exists('./'+program):
        print(program + ' exists!')
        subprocess.call(['python', program])
        print("Finished:" + program)
    else:
        print(program+' not exists!')
