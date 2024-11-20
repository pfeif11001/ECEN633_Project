import json

def readJson(jsonFile):
    with open(jsonFile, 'r') as f:
        input = json.load(f)
        return input['map']

sensorData = readJson("../DataSet/PreprocessedData/intel_gfs")
numSamplesPerRev = len(sensorData[list(sensorData)[0]]['range'])  # Get how many points per revolution
initXY = sensorData[sorted(sensorData.keys())[0]]

sensorData = {key: sensorData[key] for key in sorted(sensorData.keys())}


print(sensorData)