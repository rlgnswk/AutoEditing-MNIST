from matplotlib import pyplot as plt
import os


def loss_plot(count ,acc, title):
    plt.plot(acc)
    plt.xlabel('iter')
    plt.ylabel(title)
    plt.grid(True)
    plt.show()
    return 0


def file_generator(name):
    current_path = os.getcwd()

    experiment_folder = current_path +'\\'+ "experiments" +'\\' + name
    os.mkdir(experiment_folder)

    recon_path = experiment_folder + '\\recon'
    os.mkdir(recon_path)

    recon_pair_path = experiment_folder + '\\recon_pair'
    os.mkdir(recon_pair_path)

    fake_path = experiment_folder + '\\fake'
    os.mkdir(fake_path)    

    fake_pair_path = experiment_folder + '\\fake_pair'
    os.mkdir(fake_pair_path)

    test_fake_path = experiment_folder + '\\test_fake'
    os.mkdir(test_fake_path)

    test_fake_pair_path = experiment_folder + '\\test_fake_pair'
    os.mkdir(test_fake_pair_path)


    return recon_path, recon_pair_path, fake_path, fake_pair_path, test_fake_path, test_fake_pair_path
