import os

current_dir = os.path.dirname(os.path.abspath(__file__))

print(os.path.join(os.getcwd(), "replay.txt"),)

CONFIG = {       
	"interval": 1.0,
	"seed": 1,
	"dir": os.path.join(current_dir, "data-hangzhou_1x1_bc-tyc_18041608_1/"),
	"roadnetFile": "roadnet.json",
	"flowFile": "flow.json",
	"rlTrafficLight": True,
	"saveReplay": True,
	"roadnetLogFile": "roadnetlog.json",
	"replayLogFile": "replay.txt",
}
