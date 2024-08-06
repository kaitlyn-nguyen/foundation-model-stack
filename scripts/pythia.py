import matplotlib.pyplot as plt

# Data
processes_large = ['Compiled Forward Call', 'Export Time', 'Load Time']
times_large = [25482.427734375, 114929.4453125, 54222.84375]

processes_small = ['Normal Forward Call', 'Forward Call with Loaded Model']
times_small = [225.2728271484375, 90.77376174926758]

# # Create bar chart for larger values
# plt.figure(figsize=(10, 6))
# plt.bar(processes_large, times_large, color=['green', 'orange', 'red'])
# plt.title('Comparison of Forward Call Times (Large Values)')
# plt.xlabel('Process Type')
# plt.ylabel('Time (ms)')
# plt.show()




# Example data
processes_small = ['Eager Mode Forward Call', 'Forward Call with Loaded Model']
times_small = [225.27, 90.77]

# Define custom colors
pastel_blue = '#a3c9f7'  # Pastel blue
dark_blue = '#0f4c81'    # Dark blue

# Create bar chart for smaller values with thinner bars and reduced spacing
plt.figure(figsize=(8, 6))
bar_width = 0.1
bar_positions = [0, 0.2]  # Positions for the bars, adjusted to reduce spacing
plt.bar(bar_positions, times_small, color=[pastel_blue, dark_blue], width=bar_width)
plt.xticks(bar_positions, processes_small)
plt.title('Comparison of Forward Call Times')
plt.xlabel('Mode')
plt.ylabel('Time (ms)')
plt.show()
