% BPSK Modulation and Demodulation Simulation with BER Analysis
clear all;
close all;
clc;

% Simulation Parameters
N = 10000;          % Number of bits
Fs = 100;           % Sampling frequency
fc = 10;            % Carrier frequency
T = 1;              % Bit period
t = 0:1/Fs:T-1/Fs;  % Time vector for one bit
Eb = 1;             % Energy per bit

% Transmitter
% Generate random binary stream
data = randi([0 1], 1, N);

% Convert binary 0/1 to antipodal -1/+1
baseband_signal = 2*data - 1;

% Generate carrier signal
carrier = cos(2*pi*fc*t);

% Modulation
modulated_signal = [];
for i = 1:N
    modulated_bit = baseband_signal(i) * carrier;
    modulated_signal = [modulated_signal modulated_bit];
end

% Time vector for entire signal
t_total = 0:1/Fs:(N*T-1/Fs);

% Plotting
% Constellation Diagram
figure(1);
scatter(real(baseband_signal), imag(baseband_signal), 'filled');
title('BPSK Constellation Diagram');
xlabel('In-phase');
ylabel('Quadrature');
grid on;

% Baseband and Passband Signals
figure(2);
subplot(2,1,1);
stem(data(1:20), 'LineWidth', 2);
title('Original Binary Data (First 20 bits)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t_total(1:200), modulated_signal(1:200), 'LineWidth', 2);
title('BPSK Modulated Signal (First 2 bits)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% Spectrum Analysis
figure(3);

[pxx, f] = pwelch(modulated_signal, [], [], [], Fs);
plot(f, 10*log10(pxx), 'LineWidth', 2);
title('Power Spectral Density of BPSK Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;


% BER Analysis
EbN0dB = 0:2:20;  % Eb/N0 range in dB
EbN0 = 10.^(EbN0dB/10);  % Convert to linear scale
BER_theoretical = qfunc(sqrt(2*EbN0));  % Theoretical BER
BER_simulated = zeros(size(EbN0));

for i = 1:length(EbN0)
    % Generate noise
    noise_variance = 1/(2*EbN0(i));
    noise = sqrt(noise_variance) * randn(size(modulated_signal));
    
    % Add noise to signal
    received_signal = modulated_signal + noise;
    
    % Demodulation (Coherent Detection)
    demodulated_signal = [];
    for j = 1:N
        bit_segment = received_signal((j-1)*length(t)+1:j*length(t));
        demodulated_bit = sum(bit_segment .* carrier) > 0;
        demodulated_signal = [demodulated_signal demodulated_bit];
    end
    
    % Calculate BER
    errors = sum(data ~= demodulated_signal);
    BER_simulated(i) = errors/N;
end

% Plot BER curves
figure(4);
semilogy(EbN0dB, BER_theoretical, 'b-', 'LineWidth', 2);
hold on;
semilogy(EbN0dB, BER_simulated, 'r*-', 'LineWidth', 2);
grid on;
title('BER Performance of BPSK');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate');
legend('Theoretical', 'Simulated');
