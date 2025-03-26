from toolkit import DataParser
import matplotlib.pyplot as plt

def histogram(data):
    fig_1, axs_1 = plt.subplots(3, 10, figsize=(30,34))
    fig_2, axs_2 = plt.subplots(3, 10, figsize=(30,34))
    fig_1.suptitle("B")
    fig_2.suptitle("M")
    x = 1
    B_data = data[data.iloc[:,0] == "B"]
    M_data = data[data.iloc[:,0] == "M"]
    for i in range(0, 3):
        for j in range(0, 10):
            axs_1[i, j].hist(B_data.iloc[:, x])
            x += 1
    x = 1
    for y in range(0, 3):
        for z in range(0, 10):
            axs_2[y, z].hist(M_data.iloc[:, x])
            x += 1
    plt.show()

def describe(data):
    B_data = data[data.iloc[:,0] == "B"]
    M_data = data[data.iloc[:,0] == "M"]
    print(data.describe())
    print(B_data.describe())
    print(M_data.describe())

def main():
    data = DataParser.open_file("data.csv", 0)
    describe(data)
    histogram(data)

if __name__ == "__main__":
    main()