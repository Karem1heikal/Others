clc
clear all
close all
fc = 3400; %freq. of carrier signal
Fs = 480000; %sampling frequency
dt = 1/Fs;
m=audioread('sound.wav') ;
t =[0:size(m)-1/length(m)]*dt;

%================"Note" please
%I have tried to implement the modulation without using ready
% functions but it shows error "it is require larger memory space on my pc"
%m = cos(2*pi*fm*t);
%modulated_signal1 = cos(2*pi*fc*t + kf1*cumsum(m)/Fs);
% m = cos(2*pi*fm*t);
% integration of massage
% m_b = tril(ones(length(m)));
% m_c = m.*m_b;
% sum_m = sum(m_c,2);
%s = a*cos(2*pi*fc*t+(kf1*2*pi*sum_m').*dt);

%beta = 3 
fm = 3000; %freq. of message signal
delta_f = 9000;
B1 = delta_f/fm; %frequency sensitivity of the modulator / modulation index


modulated_signal1=modulate(m,fc,Fs,'fm',B1);

demodSig1=demod(modulated_signal1,fc,Fs,'fm',B1);

%beta = 5 
delta_f = 3*5000;
B2 = delta_f/fm; %frequency sensitivity of the modulator / modulation index
modulated_signal2=modulate(m,fc,Fs,'fm',B2);
demodSig2=demod(modulated_signal1,fc,Fs,'fm',B2);

% add wide guessian noise B = 3  
SNR = 50;
SignalWithNoiseM = awgn(m,SNR);
SNR = 50;
SignalWithNoiseS1 = awgn(modulated_signal1,SNR);

% add wide guessian noise B = 3 
SNR = 50;
SignalWithNoiseS2 = awgn(modulated_signal2,SNR);
figure(3)
subplot(3,1,1)
plot(t,SignalWithNoiseM)
xlabel ('Time(s)');
ylabel ('Amplitude');
xlim([0 1])
ylim([-1 1])
title('Message Signal Noise')
subplot(3,1,2)
plot(t,SignalWithNoiseS1)
xlabel ('Time(s)');
ylabel ('Amplitude');
xlim([0 0.05])
ylim([-5 5])
title('modulated Signal Noise B= 3')
subplot(3,1,3)
plot(t,SignalWithNoiseS2)
xlabel ('Time(s)');
ylabel ('Amplitude');
xlim([0 0.05])
ylim([-5 5])
title('modulated Signal Noise B= 5')

figure(1)
subplot(3,2,1)
plot(t,m)
xlabel ('Time(s)');
ylabel ('Amplitude');
title('Message Signal')

subplot(3,2,3)
plot(t,modulated_signal1);
xlim([0 0.05])
ylim([-5 5])
xlabel ('Time(s)');
ylabel ('Amplitude');
title('FM Modulated Signal B=3')

subplot(3,2,4)
plot(t,modulated_signal2);
xlim([0 .05])
ylim([-5 5])
xlabel ('Time(s)');
ylabel ('Amplitude');
title('FM Modulated Signal  B=5')

subplot(3,2,6)
plot(t,demodSig2);
xlim([0 1])
ylim([-1 1])
xlabel ('Time(s)');
ylabel ('Amplitude');
title('FM Modulated Signal  B=5')


subplot(3,2,5)
plot(t,demodSig1);
xlabel ('Time(s)');
ylabel ('Amplitude');
title('FM deModulated Signal  B=3')

% Spectrum calculation
N = length(t);
Lfft = 2^ceil(log2(N));
f =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));
M = fftshift(fft(m,Lfft));
S = fftshift(fft(modulated_signal1,Lfft));
R_flt = fftshift(fft(demodSig1,Lfft));


S2 = fftshift(fft(modulated_signal2,Lfft));
R_flt2 = fftshift(fft(demodSig2,Lfft));

% Spectrum calculation in noise 
N = length(t);
Lfft = 2^ceil(log2(N));
f =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));
M_msgNoise = fftshift(fft(SignalWithNoiseM,Lfft));
S1_Noise = fftshift(fft(SignalWithNoiseS1,Lfft));
S2_Noise = fftshift(fft(SignalWithNoiseS2,Lfft));

figure(4)
subplot(3,1,1)
plot(f,abs(M_msgNoise)/Fs);
title('Frequency Domain Representation of the Noisy Signals ')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-100000 100000])
grid on
subplot(3,1,2)
plot(f,abs(S1_Noise)/Fs);
title('Freq spectrum Fm modulation of Noisy Signals B = 3')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-5000 5000])

subplot(3,1,3)
plot(f,abs(S2_Noise)/Fs);
title('Freq spectrum Fm modulation of Noisy Signals B = 5')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-5000 5000])





figure(2)
subplot(3,2,1)
plot(f,abs(M)/Fs);
title('Frequency Domain Representation of the Signals')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-100000 100000])
grid on
subplot(3,2,3)
plot(f,abs(S)/Fs);
title('Freq spectrum Fm modulation of Signals B = 3')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-5000 5000])


subplot(3,2,4)
plot(f,abs(S2)/Fs);
title('Freq spectrum Fm modulation of Signals B = 5')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-5000 5000])


subplot(3,2,5)
plot(f,abs(R_flt)/Fs);
title('Frequ of the Demodulated Signal B = 3 ')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-1000 1000])

subplot(3,2,6)
plot(f,abs(R_flt2)/Fs);
title('Frequ of the Demodulated Signal B = 5 ')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-1000 1000])
