import pandas as pd
import matplotlib.pyplot as plt



FILE_PATH = "detour.csv"

trajectory = {}
obstacle = {"x":[],"y":[]}
df = pd.read_excel(FILE_PATH)

for index in df["Detour number"]:
    trajectory.update({index:{"x":[],"y":[]}})

for (index , coordinates) in zip(df["Detour number"],df["centroid"]):
    coordinates = coordinates.replace("(","")
    coordinates = coordinates.replace(")","")
    coordinates = coordinates.split(",")
    trajectory[index]["x"].append(float(coordinates[0]))
    trajectory[index]["y"].append(float(coordinates[1]))

for alex in df["Obstacle"] :
    alex = alex.replace("(","")
    alex = alex.replace(")","")
    alex = alex.split(",")
    obstacle["x"].append(float(alex[0]))
    obstacle["y"].append(float(alex[1]))

for plot in trajectory.items():
    plt.plot(plot[1]["x"] , plot[1]["y"] ,label=f"{plot[0]}") 

plt.plot(obstacle["x"],obstacle["y"],label="obstacle")

plt.gca().invert_yaxis()
plt.legend()
plt.savefig("trajectories.png",dpi=300)