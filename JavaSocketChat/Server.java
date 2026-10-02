package JavaSocketChat;

import java.io.*;
import java.net.*;

public class Server {

    public static void main(String[] args) {

        int port = 5000;

        try {
            ServerSocket serverSocket = new ServerSocket(port);

            System.out.println("Server started...");
            System.out.println("Waiting for client connection...");

            Socket socket = serverSocket.accept();

            System.out.println("Client connected!");

            BufferedReader input = new BufferedReader(
                    new InputStreamReader(socket.getInputStream()));

            PrintWriter output = new PrintWriter(
                    socket.getOutputStream(), true);

            String message = input.readLine();

            System.out.println("Client: " + message);

            output.println("Hello Client, Deepika!");

            socket.close();
            serverSocket.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}