loss = temporal_fitness_function(
    spike_times=spike_times_BC.float(),
    target_classes=y_train,
)