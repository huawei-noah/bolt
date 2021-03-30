# Setup iOS cross-compiling environment on Linux with cctools-port 

- ## Related links

  In addition to our tutorial, you can also refer to the following two links.

  - https://heroims.github.io/2017/09/10/Linux%20%E6%9E%84%E5%BB%BA:%E7%BC%96%E8%AF%91IOS:Mac%E7%A8%8B%E5%BA%8F/
  - https://medium.com/@fyf786452470/%E5%9C%A8linux%E7%9A%84%E7%9A%84%E7%9A%84%E4%B8%8B%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E7%94%9F%E6%88%90%E7%9A%84ios%E7%89%88%E7%9A%84%E5%B7%A5%E5%85%B7%E9%93%BE%E7%9A%84%E6%8C%87%E5%AF%BC%E6%89%8B%E5%86%8C-b87b472cbe14

- ## Preparations

  - llvm clang 3.9.1: You can download and install llvm clang from the [llvm website](https://releases.llvm.org/).
  - openssl 1.0.2g: Generally this tool is installed by default. **Note that if you want to copy your created iOS cross compiler toolchain to another computer for use, you need to confirm that the versions of openssl on these two machines are the same, otherwise your created toolchain can not be used.**
  - iPhoneOSSDK 10.0: If you don't have your own iPhoneOS SDK, you can download and choose one iPhoneOS SDK from [iPhoneOSSDK](https://github.com/okanon/iPhoneOS.sdk), which contains iPhoneOS SDKs from the version 8.4 to 13.2.
  - cctools 949.0.1, ld64：530: This open-source tool can help us make the ARM-iOS cross compiler toolchain and you can clone the tool from [cctools-port](https://github.com/tpoechtrager/cctools-port).

- ## Step by Step

  1. Make sure that you have available tools including llvm clang and openssl.

  2. Clone iPhoneOS SDK from [iPhoneOSSDK](https://github.com/okanon/iPhoneOS.sdk), and then place the archive **in the user home directory ~/**. For example we place it in */data/home/test*. We tried to put it in other directories, but it failed for us.

  3. Clone cctools-port from [cctools-port](https://github.com/tpoechtrager/cctools-port). 

        ```
        test@ubuntu:~$ pwd
        /data/home/test
        test@ubuntu:~$ mkdir ioscompile
        test@ubuntu:~$ cd ioscompile
        test@ubuntu:~/ioscompile$ git clone https://github.com/tpoechtrager/cctools-port.git
        test@ubuntu:~/ioscompile$ ls  
        cctools-port-master
        test@ubuntu:~$ cd ..
        ```

  4. Use the shell script build.sh of cctools-port in the directory *cctools-port-master/usage_examples/ios_toolchain/* to make aarch64/arm64-ios cross compiler toolchain. The commands are:
        ```
        test@ubuntu:~$ cd ioscompile/cctools-port-master/
        test@ubuntu:~$ ./usage_examples/ios_toolchain/build.sh  /data/home/test/iPhoneOS10.0.sdk.tar.gz arm64
        ```
        After a while, a folder **target** is created in the directory cctools-port-master/usage_examples/ios_toolchain/ and this folder **target** is the created aarch64-ios cross compiler toolchain. Now you have successfully made an ARM-IOS cross compiler toolchain on Linux. In this folder, the sub-folder */bin* contains cross compilers and related tools like *arm-apple-darwin-clang/clang++*, and the sub-folder */lib* contains the dependent libraries. By the way, if you want to make an armv7-ios cross compiler toolchain, you can change these commands like:
        ```
        test@ubuntu:~$ cd ioscompile/cctools-port-master/
        test@ubuntu:~$ ./usage_examples/ios_toolchain/build.sh  /data/home/test/iPhoneOS10.0.sdk.tar.gz armv7
        ```

  5. Configure the toolchain in your environment with the following commands or you can configure the toolchain permanently in your environment.
        ```
        test@ubuntu:~$ export PATH=/data/home/test/ioscompile/cctools-port-master/usage_examples/ios_toolchain/target/bin:$PATH
        test@ubuntu:~$ export LD_LIBRARY_PATH=/data/home/test/ioscompile/cctools-port-master/usage_examples/ios_toolchain/target/lib:$LD_LIBRARY_PATH
        ```