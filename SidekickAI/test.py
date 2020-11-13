from SidekickAI.Modules import test as moduletest
from SidekickAI.Data import test as datatest
from SidekickAI.Utilities import test as utilitiestest
import time

# Class for printing colors and bold
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

if __name__ == "__main__":
    # Run all tests
    print(color.BOLD + "|RUNNING FULL SIDEKICK AI TESTS|" + color.END)
    start_time = time.time()
    print(color.BOLD + "|TESTING MODULES|" + color.END)
    moduletest.test()
    print(color.BOLD + "|TESTING DATA|" + color.END)
    datatest.test()
    print(color.BOLD + "|TESTING UTILITIES|" + color.END)
    utilitiestest.test()
    print(color.BOLD + "Full Tests Completed in " + str(round(time.time() - start_time, 2) + "s") + color.END)