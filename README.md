# Quincunx

A [quincunx][quincunx] is a device for statistical experiments that demonstrates the
[Central Limit Theorem (CLT)][clt]. Specifically, it shows that, with sufficient sample size, the
[binomial distribution][binomial] can approximate the [normal distribution][normal].

## Running Experiments

```console
$ cargo run --release -- 11 1000 ./plot/1000.png
```

## Results

|    10 Particles     |     100 Particles     |     1000 Particles      |
| :-----------------: | :-------------------: | :---------------------: |
| ![10 Particles][10] | ![100 Particles][100] | ![1000 Particles][1000] |

Results clearly show that as the number of particles increases, the distribution gets more normal.
This is a visual proof of CLT which states that as the number of samples increases:

1. The mean gets closer to the center
2. The spread decreases
3. The distribution gets approximately normal

## License

[MIT License][license]

[quincunx]: https://en.wikipedia.org/wiki/Galton_board
[clt]: https://en.wikipedia.org/wiki/Central_limit_theorem
[binomial]: https://en.wikipedia.org/wiki/Binomial_distribution
[normal]: https://en.wikipedia.org/wiki/Normal_distribution
[10]: ./plot/10.png
[100]: ./plot/100.png
[1000]: ./plot/1000.png
[license]: LICENSE
