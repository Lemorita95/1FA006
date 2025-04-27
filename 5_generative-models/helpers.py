import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

MAIN_PATH = os.path.dirname(os.path.realpath(__file__)) # change here

def train_model(num_epochs, loader, device, latent_dimension, generator, discriminator, opt_generator, opt_discriminator, criterion, logStep):
    # tensorboard writer
    writer = SummaryWriter("runs/gan_training")

    # fixed noise for visualizing generator progress
    fixed_noise = torch.randn(64, latent_dimension, 1, 1).to(device)

    step = 0
    print("Started Training and Visualization...")
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)  # move real images to device
            batch_size = real.size(0)

            ''' Train Discriminator '''
            # labels for real and fake images
            label_real = torch.ones(batch_size, 1).to(device).view(-1)
            label_fake = torch.zeros(batch_size, 1).to(device).view(-1)

            # forward pass for real images
            output_real = discriminator(real).view(-1)
            loss_real = criterion(output_real, label_real)

            # generate fake images
            noise = torch.randn(batch_size, latent_dimension, 1, 1).to(device)
            fake = generator(noise)

            # forward pass for fake images
            output_fake = discriminator(fake.detach()).view(-1)  # avoid computing gradients for generator
            loss_fake = criterion(output_fake, label_fake)

            # total discriminator loss
            loss_discriminator = (loss_real + loss_fake) / 2

            # backpropagation and optimization for discriminator
            opt_discriminator.zero_grad()
            loss_discriminator.backward()
            opt_discriminator.step()

            ''' Train Generator '''
            # labels for generator training
            label_gen = torch.ones(batch_size, 1).to(device).view(-1)  # we want fake images to be classified as real)

            # forward pass for fake images
            output_fake_for_gen = discriminator(fake).view(-1)
            loss_generator = criterion(output_fake_for_gen, label_gen)

            # backpropagation and optimization for generator
            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            ''' Logging and Visualization '''
            if batch_idx % logStep == 0:
                print(f"\rEpoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                      f"Loss D: {loss_discriminator:.4f}, Loss G: {loss_generator:.4f}", end="")

                # add losses to TensorBoard
                writer.add_scalar("Loss/Discriminator", loss_discriminator.item(), global_step=step)
                writer.add_scalar("Loss/Generator", loss_generator.item(), global_step=step)

                # add generated images to TensorBoard
                with torch.no_grad():
                    fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake_images, normalize=True)
                    real_images = real.reshape(-1, 1, 28, 28)
                    img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)

                    writer.add_image("Generated Images", img_grid_fake, global_step=step)
                    writer.add_image("Real Images", img_grid_real, global_step=step)

                step += 1

    # close TensorBoard writer
    writer.close()