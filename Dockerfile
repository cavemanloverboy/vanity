FROM rust:latest

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/cavemanloverboy/vanity.git .
RUN cargo build --release

ENTRYPOINT ["./target/release/vanity"]
CMD ["--help"]