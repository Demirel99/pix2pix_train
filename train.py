# train.py
import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import util  # <-- Import util functions
import numpy as np    # <-- Import numpy for image concatenation

if __name__ == '__main__':
    # --- 1. Get Training Options ---
    opt = TrainOptions().parse()

    # --- 2. Create Dataset ---
    dataset, dataloader = create_dataset(opt) # <-- Unpack both
    dataset_size = len(dataset) # <-- Get length from the dataset, NOT the dataloader
    print('The number of training images = %d' % dataset_size)

    # --- 3. Create Model ---
    model = create_model(opt)
    model.setup(opt)
    total_iters = 0

    # --- Pre-training: Save data samples for verification ---
    print('Saving initial data samples for verification...')
    sample_dir = os.path.join(opt.checkpoints_dir, opt.name, 'data_samples')
    util.mkdirs(sample_dir)
    num_samples_to_save = 5
    for i in range(min(num_samples_to_save, dataset_size)):
        sample = dataset[i]
        # unsqueeze to add a batch dimension, as tensor2im expects it
        input_img_tensor = sample['A'].unsqueeze(0)
        gt_img_tensor = sample['B'].unsqueeze(0)

        input_img_np = util.tensor2im(input_img_tensor)
        gt_img_np = util.tensor2im(gt_img_tensor)

        util.save_image(input_img_np, os.path.join(sample_dir, f'sample_{i:02d}_input.png'))
        util.save_image(gt_img_np, os.path.join(sample_dir, f'sample_{i:02d}_GT.png'))
    print(f'... {min(num_samples_to_save, dataset_size)} input/GT pairs saved to {sample_dir}')

    # --- Create directory for epoch visualizations ---
    visual_dir = os.path.join(opt.checkpoints_dir, opt.name, 'epoch_visuals')
    util.mkdirs(visual_dir)
    print(f'Epoch visualizations will be saved to {visual_dir}')


    # --- 4. Training Loop ---
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # Use the dataloader for iteration
        for i, data in enumerate(dataloader): # <-- Iterate using the dataloader
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # --- Print Losses ---
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)

            iter_data_time = time.time()
            
        # --- Save Model ---
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # --- Save visualization of the last batch from the epoch ---
        print('saving visualization at the end of epoch %d' % epoch)
        visuals = model.get_current_visuals()
        # Convert tensors to numpy images
        real_A_img = util.tensor2im(visuals['real_A']) # Input
        real_B_img = util.tensor2im(visuals['real_B']) # Ground Truth
        fake_B_img = util.tensor2im(visuals['fake_B']) # Predicted Output
        # Concatenate images: Input | Ground Truth | Prediction
        combined_img = np.hstack([real_A_img, real_B_img, fake_B_img])
        save_path = os.path.join(visual_dir, f'epoch_{epoch:03d}.png')
        util.save_image(combined_img, save_path)

        # --- Print Epoch Summary ---
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()