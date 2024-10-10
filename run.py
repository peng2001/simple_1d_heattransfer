import toml
import model
import matplotlib.pyplot as plt
import numpy as np

def config_import(): 
    config_file = "config.toml"
    with open(config_file, 'r') as f:
        inputs = toml.load(f)
    return inputs

def graphing_3d(temperatures_list, time_record, cells_list):
    x = np.array(cells_list)
    t = np.array(time_record)
    temperature_data = np.array(temperatures_list)
    x, t = np.meshgrid(x, t)
    # Flatten the data for plotting
    # Plotting the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, t, temperature_data, cmap='viridis')
    # Adding axis labels
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Temperature')
    plt.show()
    ax.set_title('1D Temperature')


if __name__ == "__main__":
    inputs = config_import()
    time_record, cells_list, cells_temperatures_init = model.model_setup(dx = inputs["dx"], dt = inputs["dt"], total_time = inputs["total_time"], temperature_initial = inputs["Temperature_initial"]+273.15, thickness = inputs["Thickness"])
    temperatures = model.run_model(time_record=time_record, cells_list=cells_list, cells_temperatures_init=cells_temperatures_init,
                                    Temperature_side_minusL=inputs["Temperature_side_minusL"]+273.15, Temperature_side_plusL=inputs["Temperature_side_plusL"]+273.15, 
                                    dt=inputs["dt"], dx=inputs["dx"], Heat_gen=inputs["Heat_gen"], k=inputs["Conductivity"], rho=inputs["Density"], cp=inputs["Specific_heat"])
    temperatures_C = [[value - 273.15 for value in sublist] for sublist in temperatures] # subtract 273.15 from all points to convert to deg C
    graphing_3d(temperatures_C, time_record, cells_list)
