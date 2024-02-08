import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import sklearn.metrics
from anfis import ANFIS
import genetic_algorithm as ga


def create_initial_doe(n, bounds):
    """Create initial design of experiments with QMC Sobol sequence"""
    sampler = qmc.Sobol(d=2, scramble=False)
    points = sampler.random(n) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return points


def simulate_response(loc):
    """Simulator response at a location - replace with reservoir simulator"""
    return -np.sum(loc ** 2)  # simple quadratic response


def train_anfis(inputs, outputs):
    """Train ANFIS model on input/output data"""
    model = ANFIS()
    model.fit(inputs, outputs)
    return model


def optimize_anfis(model, bounds):
    """Optimize trained ANFIS model"""

    def objective(x):
        return model.predict([x])[0]

    opt = ga.run_ga(fitness_fn=objective, dimensions=len(bounds), bounds=bounds)
    return opt


def update_doe(point, n, bounds):
    """Add point and generate new points around it with QMC"""
    points = np.vstack((point, create_initial_doe(n, bounds)))
    return points


def asrm(max_iter, bounds):
    n = 10  # initial samples
    points = create_initial_doe(n, bounds)

    outputs = [simulate_response(p) for p in points]

    model = train_anfis(points, outputs)

    for i in range(max_iter):

        opt = optimize_anfis(model, bounds)
        true_opt = simulate_response(opt)

        if true_opt > np.max(outputs):
            points = update_doe(opt, 4, bounds)
            outputs += [simulate_response(p) for p in points[-4:]]

        model = train_anfis(points, outputs)

    return points[np.argmax(outputs)], np.max(outputs)


if __name__ == "__main__":
    bounds = np.array([[0, 5], [0, 5]])
    optimum, max_output = asrm(20, bounds)

    print(f"Optimized location: {optimum}")
    print(f"Maximum response: {max_output}")

    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = - (X ** 2 + Y ** 2)

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    ax.scatter(optimum[0], optimum[1], marker='x', s=200, color='black')
    ax.set(xlabel='x', ylabel='y')
    ax.set_title('Optimized Well Placement')
    plt.show()