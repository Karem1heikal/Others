
clc
clear all
close all

fm = 100;            %message frequency in Hz

Fs = 480000;          %sampling frequency. not to be confused with "sampling" in DSP.
dt = 1/Fs;          %sample period.

%m = sin(2*pi*fm*t); %message signal
fc= 3400;           %carrier frequency in Hz
a = 2;              %Amplitude of carrier

m=audioread('sound.wav') ;
%filter signal beyond 3.4 KHz
[b,a] = butter(6,fc/(Fs)); 
filteredSignal = filter(b, a, m);
t =[0:size(m)-1/length(m)]*dt;     %time interval

subplot (4,1,1);
plot (t,m(1:length(t)));
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Message Signal');
grid on
% hilbart transform 
mh = imag(hilbert(m));
subplot (4,1,2);
plot (t,mh);
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Hilber Transform of Message Signal');
grid on
% ssb_sc modeulation 
%================"Note" please
%I have tried to implement the modulation without using ready
% functions but it shows error "it requires larger memory space on my pc"
%s = m.*cos(2*pi*fc*t) + mh.*sin(2*pi*fc*t);
ssb_sc_signal = modulate(filteredSignal,fc,Fs,'amssb');


subplot (4,1,3);
plot (t,(ssb_sc_signal(1:length(t))));
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Modulated Signal');
grid on
% spectrum calc
N = length(t);
Lfft = 2^ceil(log2(N));
M = fftshift(fft(m,Lfft));
MH = fftshift(fft(mh,Lfft));
S = fftshift(fft(ssb_sc_signal,Lfft));
f =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));

figure
subplot(3,1,1)
plot(f,abs(M)/Fs);
title('Freq. Spectrum of Message Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
subplot(3,1,2)
plot(f,abs(S)/Fs);
title('Freq. Spectrum of Modulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
demodulated_ssb=demod(ssb_sc_signal,fc,Fs,'amssb');
r_lo = demodulated_ssb;
%r_lo = s.*sin(2*pi*fc*t);
[b,a] = butter(10,2*fc/Fs);
r_flt = filter(b,a,r_lo);
R_flt = fftshift(fft(r_flt,Lfft));
subplot(3,1,3)
plot(f,abs(R_flt)/Fs);
title('Freq. Spectrum of Demodulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on

figure(1)
subplot (4,1,4);
plot(t,r_flt)
xlabel('Time(s)')
ylabel('Amplitude')
title('Demodulated Signal')
grid on

