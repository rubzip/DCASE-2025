import torch
import torchaudio
import torchaudio.transforms as T


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        orig_sample_rate=44_100,
        sample_rate=32_000,
        n_fft=1024,
        hop_length=323, # 320
        win_length=800,
        n_mels=128,
        window_fn=torch.hann_window,
        freqm=12,
        timem=12,
    ):
        super().__init__()
        self.resampler = (
            T.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
            if orig_sample_rate != sample_rate
            else torch.nn.Identity()
        )

        self.get_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=window_fn,
            power=2.0,
        )

        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(timem, iid_masks=True),
        )

    def forward(self, waveform, augment=False):
        w_resampled = self.resampler(waveform)
        mel = self.get_spectrogram(w_resampled)
        log_mel = self.amplitude_to_db(mel)
        if augment:
            log_mel = self.mel_augment(log_mel)
#        if augment:
#            mel = self.mel_augment(mel)
#        log_mel = self.amplitude_to_db(mel)
        return log_mel