
% AM Mode
clc
clear all
close all

fm = 100;            %message frequency in Hz

Fs = 480000;          %sampling frequency. not to be confused with "sampling" in DSP.
dt = 1/Fs;          %sample period.

%m = sin(2*pi*fm*t); %message signal
fc= 3400;           %carrier frequency in Hz
a = 2;              %Amplitude of carrier
mod_index = 0.8;    %modulation index
m=audioread('sound.wav') ;
letter=audioread('voice letter.wav');
t =[0:size(m)-1/length(m)]*dt;     %time interval

% am modulation

%================"Note" please
%I have tried to implement the modulation without using ready
% functions but it shows error "it is required larger memory space on my pc"
s = modulate(m,fc,Fs,'am',mod_index);
%s = a*(1 + (mod_index.*m(:,1))).*cos(2*pi*fc*t); %modulated signal
%filter signal beyond 3.4 KHz
[b,a] = butter(6,fc/(Fs)); 
filteredSignal = filter(b, a, m);

[b,a] = butter(6,fc/(Fs)); 
filteredChrSignal = filter(b, a, letter);

s2 = modulate(filteredChrSignal,fc,Fs,'am',mod_index);
% spectrum calculation letters
N = length(t);
Lfft = 2^ceil(log2(N));
M2 = fftshift(fft(letter,Lfft));
S2 = fftshift(fft(s2,Lfft));
f2 =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));
% spectrum calculation
N = length(t);
Lfft = 2^ceil(log2(N));
M = fftshift(fft(m,Lfft));
S = fftshift(fft(s,Lfft));
f =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));
% plot Spectrum
figure
subplot(2,3,1)
plot(f,abs(M)/Fs);
title('Freq. Spectrum of Message')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
subplot(2,3,2)
plot(f,abs(S)/Fs);
title('Freq. Spectrum of Modulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
% envelope detector
r_env = abs(s);
r_env2 = abs(s2);
% filter calc 
[b,a] = butter(5,2*fc/Fs);
r_flt = filter(b,a,r_env);
R_flt = fftshift(fft(r_flt,Lfft));
% filter calc 2
[b,a] = butter(5,2*fc/Fs);
r_flt2 = filter(b,a,r_env2);
R_flt2 = fftshift(fft(r_flt,Lfft));
% plot filter Spectrum
subplot(2,3,3)
plot(f,abs(R_flt)/Fs);
title('Freq. Spectrum of Demodulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on

subplot(2,3,4)
plot(f,abs(M2)/Fs);
title('Freq. Spectrum of ChrMessage')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on

subplot(2,3,5)
plot(f2,abs(M2)/Fs);
title('Freq. Spectrum of ChrMessage')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
subplot(2,3,6)
plot(f2,abs(S2)/Fs);
title('Freq. Spectrum of Chr Modulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on

figure
subplot(4,2,4)
plot(t,r_env(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Envelope of Modulated Signal')
grid on
subplot(4,2,3)

plot(t,r_flt(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Demodulated Signal')
grid on
%let's see our message and modulated signal
subplot(4,2,1)
plot(t,m(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Message Signal')
grid on
subplot(4,2,2)
plot(t,s(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Modulated Signal')
grid on

subplot(4,2,6)
plot(t,s2(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Chr Modulated Signal')
grid on
subplot(4,2,5)
plot(t,letter(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Chr Message Signal')
grid on
subplot(4,2,8)
plot(t,r_env2(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Envelope of Chr Modulated Signal')
grid on

subplot(4,2,7)

plot(t,r_flt2(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Clr Demodulated Signal')
grid on

recorded_energy = sum(m.^2);
demodulated_energy = sum(r_flt.^2);

scaling_factor = sqrt(recorded_energy/demodulated_energy);
scaled_demodulated_signal = scaling_factor * r_flt;
figure
plot(t,scaled_demodulated_signal(1:length(t)));
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('removed(DC) DeModulated Signal')
grid on
