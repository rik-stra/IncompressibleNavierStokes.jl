using CairoMakie
using FFTW
# 1D

x = 0:0.01:0.5-0.01
y = sinpi.(2*x).*x
y_long = cat(y,[0], -1 .*y[end:-1:2], dims=1)
y_hat = fft(y_long)
y_fft_rec = real(ifft(y_hat))[1:Int(end//2)]


f = Figure()
ax1 = Axis(f[1, 1])
lines!(ax1, x, y, color = :blue)
ax2 = Axis(f[2, 1])
lines!(ax2, x, y_fft_rec, color = :red)
display(f)
lines(real(ifft(y_hat)))

k = fftfreq(length(x)*2, length(x)*2)./1

f = Figure()
y_prime = 2*pi*cospi.(2*x).*x.+sinpi.(2*x)
y_fft_prime = real(2*pi*ifft(k.*im.*y_hat))[1:Int(end//2)]
ax1 = Axis(f[1, 1])
lines!(ax1, x, y_prime, color = :blue)
ax2 = Axis(f[2, 1])
lines!(ax2, x, y_fft_prime, color = :red)

display(f)

### DCT
y_dct = dct(y)

A = zeros(Float64, 15)
DCT! = FFTW.plan_r2r!(A, FFTW.REDFT10)
A = y
y_dct2=DCT!*A

y_hat

### FFT

x = 0:0.01:0.5-0.01
y = sinpi.(2*x).*x
y_hat = fft(y)
y_fft_rec = real(ifft(y_hat))


f = Figure()
ax1 = Axis(f[1, 1])
lines!(ax1, x, y, color = :blue)
ax2 = Axis(f[2, 1])
lines!(ax2, x, y_fft_rec, color = :red)
display(f)

k = fftfreq(length(x), length(x))./1

f = Figure()
y_prime = 2*pi*cospi.(2*x).*x.+sinpi.(2*x)
y_fft_prime = real(2*pi*ifft(k.*im.*y_hat))
ax1 = Axis(f[1, 1])
lines!(ax1, x, y_prime, color = :blue)
ax2 = Axis(f[2, 1])
lines!(ax2, x, y_fft_prime, color = :red)

display(f)