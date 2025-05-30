import os
import numpy as np
import wave
import glob
import random
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm


def norm_wav(wav):
    #  norm wav value to [-1.0, 1.0]
    norm = max(np.absolute(wav))
    if norm > 1e-5:
        wav = wav / norm
    return wav, norm


def Create_data(input_dir, output_dir, trial_path, noise_list, overlap_min, overlap_max, snr_min, snr_max):
    # Get all utterance paths in Vox1 trials
    utt_paths = glob.glob(os.path.join(input_dir, '*/*/*.wav'))
    print('Sucessfully obtain all utterances from {}'.format(trial_path))
    # Get all noise paths in noise_list
    noise_lines = open(trial_path, 'r').readlines()
    print('Sucessfully obtain all noise from {}'.format(noise_list))

    # Sort the order of trials, let the test utterances in character order
    lines = open(trial_path, 'r').readlines()
    sorted_lines = sorted(lines, key=lambda line: line.split()[2])
    print('Sort the order of trials')

    print('Start creating data')
    current_utt = sorted_lines[0].strip().split()[2]
    segs_list, enroll_spk_list = [], []

    def process_seg():
        # Randomly choose one utt for test to combine
        audio1_path, audio2_path, noise_path, audio1_name, audio2_name, snr, overlap_ratio = choose_param(
            segs_list, enroll_spk_list, utt_paths, noise_lines, overlap_min, overlap_max, snr_min, snr_max)

        # Randomly change the location of test and interfere
        if random.uniform(0, 1) < 0.5:
            temp = audio1_path
            audio1_path = audio2_path
            audio2_path = temp

        # Simulate in three ways
        concat_audio, _ = concat(audio1_path, audio2_path, snr)
        mix_audio, _ = mix(audio1_path, audio2_path, snr)
        overlap_audio, _ = overlap(audio1_path, audio2_path, snr, overlap_ratio)
        noisy_audio, _ = noisy(audio1_path, noise_path, snr)

        # Save wav and trial
        save_waveform(output_dir, audio1_name, concat_audio, mix_audio, overlap_audio, noisy_audio)
        for segs in segs_list:
            write_txt(segs, output_dir, trial_path)

    count = 0

    for line in sorted_lines:
        # segs = [target/nontarget, enroll, test]
        segs = line.strip().split()
        test_utt = segs[2]
        # One test corresponds to multiple enroll, confirm that one test is processed only once
        if test_utt == current_utt:
            # Collects all pairs for the current test
            count = count + 1
            segs_list.append(segs)
            enroll_spk_list.append(segs[1].split('/')[0])
        else:
            # Process the test
            print("utt: {}, num_trials: {}".format(current_utt, count))
            process_seg()

            current_utt = test_utt
            count = 1
            segs_list, enroll_spk_list = [], []
            segs_list.append(segs)
            enroll_spk_list.append(segs[1].split('/')[0])

    # process the last test
    process_seg()



def choose_param(segs_list, enroll_spk_list, utt_paths, noise_lines, overlap_min, overlap_max, snr_min, snr_max):
    snr = random.randint(snr_min * 2, snr_max * 2) / 2
    overlap_ratio = random.uniform(overlap_min, overlap_max)
    audio1_path = os.path.join(input_dir, segs_list[-1][2])
    _, audio1 = wavfile.read(audio1_path)

    # Ensure that the selected utt length is sufficient for overlap, and not from any enroll
    choose_suitable = False
    while choose_suitable is False:
        audio2_paths = random.sample(utt_paths, 20)
        for audio2_path in audio2_paths:
            _, audio2 = wavfile.read(audio2_path)
            if audio2_path.split('/')[-3] in enroll_spk_list or audio2_path.split('/')[-3] == audio1_path.split('/')[
                -3] or len(audio1) * overlap_ratio >= len(audio2) or len(audio2) * overlap_ratio >= len(audio1):
                continue
            else:
                choose_suitable = True
                break
        if choose_suitable is False: overlap_ratio = random.uniform(overlap_min, overlap_max)

    audio1_name = audio1_path.replace(input_dir + '/', '')
    audio2_name = audio2_path.replace(input_dir + '/', '')

    noise_path = noise_lines[random.randint(0, len(noise_lines))]
    return audio1_path, audio2_path, noise_path, audio1_name, audio2_name, snr, overlap_ratio



def overlap(audio1_path, audio2_path, snr, overlap_ratio):
    _, audio1 = wavfile.read(audio1_path, "rb")
    _, audio2 = wavfile.read(audio2_path, "rb")
    audio1, _ = norm_wav(audio1.astype(np.float32) / (1 << 15))
    audio2, _ = norm_wav(audio2.astype(np.float32) / (1 << 15))

    audio1_len = audio1.shape[0]
    audio2_len = audio2.shape[0]

    mix_len = (audio1_len + audio2_len) / (1.0 + overlap_ratio)
    mix_len = int(mix_len)
    overlap_len = int(mix_len * overlap_ratio)
    audio2_len = mix_len + overlap_len - audio1_len
    audio2 = audio2[:audio2_len]

    audio1_db = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio2_db = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio2 = np.sqrt(10 ** ((audio1_db - audio2_db - snr) / 10)) * audio2

    mix_audio = np.zeros(mix_len)
    mix_audio[:audio1_len] = mix_audio[:audio1_len] + audio1
    mix_audio[-audio2_len:] = mix_audio[-audio2_len:] + audio2

    mix_audio, norm = norm_wav(mix_audio)
    return mix_audio, norm


def concat(audio1_path, audio2_path, snr):
    _, audio1 = wavfile.read(audio1_path, "rb")
    _, audio2 = wavfile.read(audio2_path, "rb")
    audio1, _ = norm_wav(audio1.astype(np.float32) / (1 << 15))
    audio2, _ = norm_wav(audio2.astype(np.float32) / (1 << 15))

    audio_db_1 = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio_db_2 = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio2 = np.sqrt(10 ** ((audio_db_1 - audio_db_2 - snr) / 10)) * audio2
    concat_audio = np.concatenate([audio1, audio2])

    concat_audio, norm = norm_wav(concat_audio)
    return concat_audio, norm


def mix(audio1_path, audio2_path, snr):
    _, audio1 = wavfile.read(audio1_path, "rb")
    _, audio2 = wavfile.read(audio2_path, "rb")
    audio1, _ = norm_wav(audio1.astype(np.float32) / (1 << 15))
    audio2, _ = norm_wav(audio2.astype(np.float32) / (1 << 15))

    # Adjust energy ratio
    audio_db_1 = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio_db_2 = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio2 = np.sqrt(10 ** ((audio_db_1 - audio_db_2 - snr) / 10)) * audio2

    # Repeat shorter audio to match the length of the longer audio
    if len(audio1) > len(audio2):
        audio2 = np.tile(audio2, (len(audio1) // len(audio2) + 1))[:len(audio1)]
    elif len(audio2) > len(audio1):
        audio1 = np.tile(audio1, (len(audio2) // len(audio1) + 1))[:len(audio2)]

    # Mix the audio
    mixed_audio = audio1 + audio2

    # Normalize the mixed audio
    mixed_audio, norm = norm_wav(mixed_audio)
    return mixed_audio, norm



def noisy(audio_path, moise_path, snr):
    _, audio1 = wavfile.read(audio_path, "rb")
    _, audio2 = wavfile.read(moise_path, "rb")
    audio1, _ = norm_wav(audio1.astype(np.float32) / (1 << 15))
    audio2, _ = norm_wav(audio2.astype(np.float32) / (1 << 15))

    # Adjust energy ratio
    audio_db_1 = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio_db_2 = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio2 = np.sqrt(10 ** ((audio_db_1 - audio_db_2 - snr) / 10)) * audio2

    # Repeat shorter audio to match the length of the longer audio
    if len(audio1) > len(audio2):
        audio2 = np.tile(audio2, (len(audio1) // len(audio2) + 1))[:len(audio1)]
    elif len(audio2) > len(audio1):
        startframe = random.randint(0, audio2.shape[0] - audio1.shape[0])
        audio2 = audio2[startframe:startframe + audio1.shape[0]]

    # Mix the audio
    mixed_audio = audio1 + audio2

    # Normalize the mixed audio
    mixed_audio, norm = norm_wav(mixed_audio)
    return mixed_audio, norm


def save_waveform(output_dir, audio1_name, concat_audio, mix_audio, overlap_audio, noisy_audio):
    for mode, audio in zip(['concat', 'mix', 'overlap', 'noisy'], [concat_audio, mix_audio, overlap_audio, noisy_audio]):
        target_output_dir = os.path.join(output_dir, mode)
        if not os.path.exists(target_output_dir):
            os.mkdir(target_output_dir)
        path = os.path.join(target_output_dir, audio1_name.replace('.wav', f'-{mode}.wav'))
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        sf.write(path, audio, 16000)


def write_txt(segs, output_dir, trial_path):
    trial_name = os.path.basename(trial_path)
    for mode in ['concat', 'mix', 'overlap', 'noisy']:
        with open(os.path.join(output_dir, trial_name.replace('.txt', f'-{mode}.txt')), "a") as t:
            t.write(segs[0] + ' ' + os.path.join(input_dir, segs[1]) + ' ' +
                    os.path.join(output_dir, segs[2].replace('.wav', f'-{mode}.wav')))
            t.write('\r\n')



if __name__ == '__main__':
    input_dir = ''  # the directory of eval data
    trial_path = '' # the path of trial_list
    noise_list = '' # the path of noise_list sampled from MUSAN
    output_dir = './data'

    overlap_min = 0.1
    overlap_max = 0.9
    snr_min = -3.0
    snr_max = 3.0
    random.seed(0)

    Create_data(input_dir, output_dir, trial_path, noise_list, overlap_min, overlap_max, snr_min, snr_max)