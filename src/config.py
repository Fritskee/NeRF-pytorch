class Config:
    # Dataset
    dataset_path = 'path/to/your/dataset'

    # Camera parameters (dummy variables)
    focal_length = (500.0, 500.0)  # (f_x, f_y)
    optical_center = (256.0, 256.0)  # (c_x, c_y)
    camera_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R
    camera_translation = [0, 0, -2]  # t

    # Image dimensions
    image_width = 512
    image_height = 512

    # Training
    batch_size = 1024
    num_epochs = 30
    learning_rate = 1e-4
    num_sample_points = 64
    optimizer = 'Adam'
    loss_function = 'mse'

    # Output
    output_dir = 'path/to/output/directory'
    model_checkpoint_dir = 'path/to/model/checkpoints'
    rendered_images_dir = 'path/to/rendered/images'
