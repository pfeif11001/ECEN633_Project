import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from OccupancyGrid import OccupancyGrid
from ScanMatcher_OGBased import ScanMatcher
import math
import copy

class ParticleFilter:
    def __init__(self, numParticles, ogParameters, smParameters):
        self.numParticles = numParticles
        self.particles = []
        self.initParticles(ogParameters, smParameters)
        self.step = 0
        self.prevMatchedReading = None
        self.prevRawReading = None
        self.particlesTrajectory = []

    def initParticles(self, ogParameters, smParameters):
        for i in range(self.numParticles):
            p = Particle(ogParameters, smParameters)
            self.particles.append(p)

    def updateParticles(self, reading, count):
        for i in range(self.numParticles):
            self.particles[i].update(reading, count)


    def weightUnbalanced(self):
        self.normalizeWeights()
        variance = 0
        for i in range(self.numParticles):
            variance += (self.particles[i].weight - 1 / self.numParticles) ** 2
            #variance += self.particles[i].weight**2
        #print(variance)
        if variance > ((self.numParticles - 1) / self.numParticles)**2 + (self.numParticles - 1.000000000000001) * (1 / self.numParticles)**2:
        #if variance > 2 / self.numParticles:
            return True
        else:
            return False

    def normalizeWeights(self):
        weightSum = 0
        for i in range(self.numParticles):
            weightSum += self.particles[i].weight
        for i in range(self.numParticles):
            self.particles[i].weight = self.particles[i].weight / weightSum

    def resample(self):
        # for particle in self.particles:
        #     particle.plotParticle()
        #     print(particle.weight)
        weights = np.zeros(self.numParticles)
        tempParticles = []
        for i in range(self.numParticles):
            weights[i] = self.particles[i].weight
            tempParticles.append(copy.deepcopy(self.particles[i]))
        resampledParticlesIdx = np.random.choice(np.arange(self.numParticles), self.numParticles, p=weights)
        for i in range(self.numParticles):
            self.particles[i] = copy.deepcopy(tempParticles[resampledParticlesIdx[i]])
            self.particles[i].weight = 1 / self.numParticles

class Particle:
    def __init__(self, ogParameters, smParameters):
        initMapXLength, initMapYLength, initXY, unitGridSize, lidarFOV, lidarMaxRange, numSamplesPerRev, wallThickness = ogParameters
        scanMatchSearchRadius, scanMatchSearchHalfRad, scanSigmaInNumGrid,  moveRSigma, maxMoveDeviation,\
        turnSigma, missMatchProbAtCoarse, coarseFactor  = smParameters
        og = OccupancyGrid(initMapXLength, initMapYLength, initXY, unitGridSize, lidarFOV, numSamplesPerRev, lidarMaxRange, wallThickness)
        sm = ScanMatcher(og, scanMatchSearchRadius, scanMatchSearchHalfRad, scanSigmaInNumGrid, moveRSigma, maxMoveDeviation, turnSigma, missMatchProbAtCoarse, coarseFactor)
        self.og = og
        self.sm = sm
        self.xTrajectory = []
        self.yTrajectory = []
        self.weight = 1

    def updateEstimatedPose(self, currentRawReading):
        estimatedTheta = self.prevMatchedReading['theta'] + currentRawReading['theta'] - self.prevRawReading['theta']
        estimatedReading = {'x': self.prevMatchedReading['x'], 'y': self.prevMatchedReading['y'], 'theta': estimatedTheta,
                            'range': currentRawReading['range']}
        dx, dy = currentRawReading['x'] - self.prevRawReading['x'], currentRawReading['y'] - self.prevRawReading['y']
        estMovingDist = math.sqrt(dx ** 2 + dy ** 2)
        rawX, rawY, prevRawX, prevRawY = currentRawReading['x'], currentRawReading['y'], self.prevRawReading['x'], \
                                         self.prevRawReading['y']
        rawXMove, rawYMove = rawX - prevRawX, rawY - prevRawY
        rawMove = math.sqrt((rawX - prevRawX) ** 2 + (rawY - prevRawY) ** 2)

        if rawMove > 0.3:
            if self.prevRawMovingTheta != None:
                if rawYMove > 0:
                    rawMovingTheta = math.acos(rawXMove / rawMove)  # between -pi and +pi
                else:
                    rawMovingTheta = -math.acos(rawXMove / rawMove)
                rawTurnTheta = rawMovingTheta - self.prevRawMovingTheta
                estMovingTheta = self.prevMatchedMovingTheta + rawTurnTheta
            else:
                if rawYMove > 0:
                    rawMovingTheta = math.acos(rawXMove / rawMove)  # between -pi and +pi
                else:
                    rawMovingTheta = -math.acos(rawXMove / rawMove)
                estMovingTheta = None
        else:
            rawMovingTheta = None
            estMovingTheta = None

        return estimatedReading, estMovingDist, estMovingTheta, rawMovingTheta

    def getMovingTheta(self, matchedReading):
        x, y, theta, range = matchedReading['x'], matchedReading['y'], matchedReading['theta'], matchedReading['range']
        prevX, prevY = self.xTrajectory[-1], self.yTrajectory[-1]
        xMove, yMove = x - prevX, y - prevY
        move = math.sqrt(xMove ** 2 + yMove ** 2)
        if move != 0:
            if yMove > 0:
                movingTheta = math.acos(xMove / move)
            else:
                movingTheta = -math.acos(xMove / move)
        else:
            movingTheta = None
        return movingTheta

    def update(self, reading, count):
        if count == 1:
            self.prevRawMovingTheta, self.prevMatchedMovingTheta = None, None
            matchedReading, confidence = reading, 1
        else:
            currentRawReading = reading
            estimatedReading, estMovingDist, estMovingTheta, rawMovingTheta = self.updateEstimatedPose(currentRawReading)
            matchedReading, confidence = self.sm.matchScan(estimatedReading, estMovingDist, estMovingTheta, count, matchMax=False)
            self.prevRawMovingTheta = rawMovingTheta
            self.prevMatchedMovingTheta = self.getMovingTheta(matchedReading)
        self.updateTrajectory(matchedReading)
        self.og.updateOccupancyGrid(matchedReading)
        self.prevMatchedReading, self.prevRawReading = matchedReading, reading
        self.weight *= confidence

    def updateTrajectory(self, matchedReading):
        x, y = matchedReading['x'], matchedReading['y']
        self.xTrajectory.append(x)
        self.yTrajectory.append(y)

    def plotParticle(self):
        plt.figure(figsize=(19.20, 19.20))
        plt.scatter(self.xTrajectory[0], self.yTrajectory[0], color='r', s=500)
        colors = iter(cm.rainbow(np.linspace(1, 0, len(self.xTrajectory) + 1)))
        for i in range(len(self.xTrajectory)):
            plt.scatter(self.xTrajectory[i], self.yTrajectory[i], color=next(colors), s=35)
        plt.scatter(self.xTrajectory[-1], self.yTrajectory[-1], color=next(colors), s=500)
        plt.plot(self.xTrajectory, self.yTrajectory)
        self.og.plotOccupancyGrid([-13, 20], [-25, 7], plotThreshold=False)
        
def readJson(jsonFile):
    with open(jsonFile, 'r') as f:
        input = json.load(f)
        return input['map']
    
def calculateAngles(numSamplesPerRev, lidarFOV):
    return np.linspace(-lidarFOV / 2, lidarFOV / 2, numSamplesPerRev).tolist()

def processSensorData(pf, sensorData, plotTrajectory=True):
    X_t, U_t, Z_tp1 = [], [], []
    
    count = 0

    plt.ion()  # Turn on interactive mode

    plt.figure(figsize=(19.20, 19.20))
    limit = 100
    for key in sorted(sensorData.keys())[:limit]:
    # for key in sorted(sensorData.keys()):

        count += 1
        pf.updateParticles(sensorData[key], count)
        if pf.weightUnbalanced():
            pf.resample()

        plt.clf()  # Clear the current figure
        maxWeight = -1
        for particle in pf.particles:
            if maxWeight < particle.weight:
                maxWeight = particle.weight
                bestParticle = particle
            plt.plot(particle.xTrajectory, particle.yTrajectory)

        xRange, yRange = [-13, 20], [-25, 7]
        ogMap = bestParticle.og.occupancyGridVisited / bestParticle.og.occupancyGridTotal
        xIdx, yIdx = bestParticle.og.convertRealXYToMapIdx(xRange, yRange)
        ogMap = ogMap[yIdx[0]: yIdx[1], xIdx[0]: xIdx[1]]
        ogMap = np.flipud(1 - ogMap)
        plt.imshow(ogMap, cmap='gray', extent=[xRange[0], xRange[1], yRange[0], yRange[1]])
        plt.draw()  # Update the plot
        plt.pause(0.01)  # Pause to allow the plot to update

        # Save data
        X_t.append((bestParticle.prevMatchedReading['x'], bestParticle.prevMatchedReading['y'], bestParticle.prevMatchedReading['theta']))
        
        # Calculate control input U_t
        if count > 1:
            prev_x, prev_y, prev_theta = X_t[-2]
            curr_x, curr_y, curr_theta = X_t[-1]
            control_input = (curr_x - prev_x, curr_y - prev_y, curr_theta - prev_theta)
            U_t.append(control_input)
        else:
            U_t.append((0, 0, 0))  # No movement for the first control input

        # Save observation Z_tp1
        Z_tp1.append(sensorData[key]['range'])

        # Save angles
        if count == 1:
            numSamplesPerRev = len(sensorData[key]['range'])
            lidarFOV = np.pi  # Assuming lidarFOV is pi, adjust if necessary
            angles = calculateAngles(numSamplesPerRev, lidarFOV)

    plt.ioff()  # Turn off interactive mode
    plt.close()
    maxWeight = 0
    for particle in pf.particles:
        # particle.plotParticle()
        if maxWeight < particle.weight:
            maxWeight = particle.weight
            bestParticle = particle
    # bestParticle.plotParticle()

    np.save('map.npy', ogMap)
        
    data = {}
    data['X_t'] = np.asarray(X_t)
    data['U_t'] = np.asarray(U_t)
    data['Z_tp1'] = np.asarray(Z_tp1)
    data['angles'] = np.asarray(angles)

    
    # save data as realWorld-dataset.npz
    np.savez('realWorld-dataset.npz', **data)

def main():
    initMapXLength, initMapYLength, unitGridSize, lidarFOV, lidarMaxRange = 50, 50, 0.02, np.pi, 10  # in Meters
    scanMatchSearchRadius, scanMatchSearchHalfRad, scanSigmaInNumGrid, wallThickness, moveRSigma, maxMoveDeviation, turnSigma, \
        missMatchProbAtCoarse, coarseFactor = 1.4, 0.25, 2, 5 * unitGridSize, 0.1, 0.25, 0.3, 0.15, 5
    sensorData = readJson("../DataSet/PreprocessedData/intel_gfs")
    numSamplesPerRev = len(sensorData[list(sensorData)[0]]['range'])  # Get how many points per revolution
    initXY = sensorData[sorted(sensorData.keys())[0]]
    numParticles = 10
    ogParameters = [initMapXLength, initMapYLength, initXY, unitGridSize, lidarFOV, lidarMaxRange, numSamplesPerRev, wallThickness]
    smParameters = [scanMatchSearchRadius, scanMatchSearchHalfRad, scanSigmaInNumGrid, moveRSigma, maxMoveDeviation, turnSigma, \
        missMatchProbAtCoarse, coarseFactor]
    pf = ParticleFilter(numParticles, ogParameters, smParameters)
    processSensorData(pf, sensorData, plotTrajectory=True)

if __name__ == '__main__':
    main()