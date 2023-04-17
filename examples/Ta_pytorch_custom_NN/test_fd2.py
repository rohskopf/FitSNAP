from fitsnap3lib.tools.test_tools import TestTools
from fitsnap3lib.io.input import Config
input_script = "Ta-example.in"
test_tool = TestTools(input_script)
test_tool.finite_difference(group="Displaced_BCC")