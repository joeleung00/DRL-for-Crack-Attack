# import time
# import sys
# for i in range(5):
#     print (str(i) + "hello")
#     time.sleep(0.5)

# time.sleep(3)
# for i in range(5):
#     print (str(i) + "bye")
#     time.sleep(1)
from pynput.keyboard import Key, Controller
import subprocess
import time
keyboard = Controller()
subprocess.Popen(["python3", "../game_simulation/Game.py", "ai"])
time.sleep(5)
keyboard.press("r")
keyboard.release("r")
