# train.py
import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

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

        # --- Print Epoch Summary ---
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()