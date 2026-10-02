# Java Client-Server Communication Using Socket Programming

## Project Overview

This project demonstrates client-server communication using Java Socket Programming.

The server listens for client connections using `ServerSocket`, while the client connects to the server using `Socket`.

## Technologies Used

* Java
* Socket Programming
* TCP/IP
* VS Code
* Git & GitHub

## Project Structure

```text
JavaSocketChat/
│
├── Server.java
├── Client.java
├── README.md
└── .gitignore
```

## How It Works

The server runs on one laptop and listens on port `5000`.

The client runs on another laptop and connects to the server using the server laptop's IPv4 address.

Both laptops should be connected to the same network.

## How to Run

### Start Server

```bash
javac Server.java
java Server
```

### Start Client

First update the server IP address in `Client.java`:

```java
String serverIP = "SERVER_IP_ADDRESS";
```

Then run:

```bash
javac Client.java
java Client
```

## Example

```text
Server:
Waiting for client...
Client connected!
Client: Hello Server

Client:
Connected to server!
Server: Hello Client
```

## Networking

Example:

```text
Server IP: 192.168.43.25
Port: 5000
```

The client connects to:

```text
192.168.43.25:5000
```

## Learning Outcomes

* Understanding client-server architecture
* Java Socket and ServerSocket
* TCP communication
* IP addresses and ports
* Network communication between two computers
* Basic Git and GitHub workflow
