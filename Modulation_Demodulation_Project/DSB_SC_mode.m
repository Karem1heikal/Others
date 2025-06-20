% DSB_SC Mode
clc
clear all
close all
% fm = 100;            %message frequency in Hz
% Fs = 48000;          %sampling frequency. not to be confused with "sampling" in DSP.
% dt = 1/Fs;          %sample period. 
% t = (0:dt:1);      %time interval
% fc= Fs/10;           %carrier frequency in Hz
% a = 2;              %Amplitude of carrier
% [m,Fs]=audioread('C:\Users\user\Downloads\sound.wav') ;


fm = 100;            %message frequency in Hz

Fs = 480000;          %sampling frequency. not to be confused with "sampling" in DSP.
dt = 1/Fs;          %sample period.

%m = sin(2*pi*fm*t); %message signal
fc= 3400;           %carrier frequency in Hz
a = 2;              %Amplitude of carrier
m=audioread('sound.wav') ;

t =[0:size(m)-1/length(m)]*dt;     %time interval

%filter signal beyond 3.4 KHz
[b,a] = butter(6,fc/(Fs)); 
filteredSignal = filter(b, a, m);

% modulation
%s = 2.*filteredSignal.*sin (2*pi*fc*t);
%filter signal beyond 3.4 KHz
s = modulate(m,fc,Fs,'amdsb-tc',1);


subplot (4,1,1);
plot (t,m);
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Message Signal');
grid on
c = 2.*sin (2*pi*fc*t);
subplot (4,1,2);
plot (t,c);
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Carrier Signal');
grid on

% modulation
%s = 2.*filteredSignal.*sin (2*pi*fc*t);
subplot (4,1,3);
plot (t,s);
xlabel ('Time(s)');
ylabel ('Amplitude');
title ('Modulated Signal');
grid on

% spectrum calculation
N = length(t);
Lfft = 2^ceil(log2(N));
M = fftshift(fft(m,Lfft));
C = fftshift(fft(c,Lfft));
S = fftshift(fft(s,Lfft));
f =(-Lfft/2:Lfft/2-1)/(Lfft*(1/Fs));
figure(2);
subplot(4,1,1)
plot(f,abs(M)/Fs);
title('Freq. Spectrum of Message Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
subplot(4,1,2)
plot(f,abs(C)/Fs);
title('Freq. Spectrum of Carrier Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
subplot(4,1,3)
plot(f,abs(S)/Fs);
title('Freq. Spectrum of Modulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
% coherant detector
r_lo=demod(s,fc,Fs,'am',1);
%r_lo = s.*c;
% Low pass filter
[b,a] = butter(5,2*fc/Fs);
r_flt = filter(b,a,r_lo);
R_flt = fftshift(fft(r_flt,Lfft));
subplot(4,1,4)
plot(f,abs(R_flt)/Fs);
title('Freq. Spectrum of Demodulated Signal')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
grid on
figure(1);
subplot (4,1,4);
plot(t,r_flt)
xlabel('Time(s)')
ylabel('Amplitude(v)')
title('Demodulated Signal')
grid on
