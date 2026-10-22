# Attack-family taxonomy

Attack types mapped to `other` (5): `Analysis`, `Attack`, `Fuzzers`, `Generic`, `mitm`

| attack_type | family | defence | datasets | rows | in_per_attack |
|---|---|---|---|---|---|
| Benign | benign |  | cicids2018, iot23 | 105346 | False |
| BENIGN | benign |  | cicids2017, usbids1 | 91727 | False |
| normal | benign |  | ctu13, toniot | 65915 | False |
| Normal | benign |  | insdn, unswnb15 | 83286 | False |
| Bot | botnet |  | cicids2017, cicids2018 | 2908 | True |
| BOTNET | botnet |  | insdn | 101 | True |
| botnet-s42 | botnet |  | ctu13 | 8228 | False |
| botnet-s43 | botnet |  | ctu13 | 4064 | False |
| botnet-s44 | botnet |  | ctu13 | 1564 | False |
| botnet-s45 | botnet |  | ctu13 | 202 | False |
| botnet-s46 | botnet |  | ctu13 | 225 | False |
| botnet-s47 | botnet |  | ctu13 | 501 | False |
| botnet-s48 | botnet |  | ctu13 | 17 | False |
| botnet-s49 | botnet |  | ctu13 | 328 | False |
| botnet-s50 | botnet |  | ctu13 | 34448 | False |
| botnet-s51 | botnet |  | ctu13 | 2603 | False |
| botnet-s52 | botnet |  | ctu13 | 187 | False |
| botnet-s53 | botnet |  | ctu13 | 520 | False |
| botnet-s54 | botnet |  | ctu13 | 4067 | False |
| C&C | botnet |  | iot23 | 4291 | True |
| C&C-FileDownload | botnet |  | iot23 | 46 | True |
| C&C-HeartBeat | botnet |  | iot23 | 556 | True |
| C&C-HeartBeat-FileDownload | botnet |  | iot23 | 8 | True |
| C&C-Mirai | botnet |  | iot23 | 1 | True |
| C&C-Torii | botnet |  | iot23 | 17 | True |
| FileDownload | botnet |  | iot23 | 13 | True |
| Okiru | botnet |  | iot23 | 813 | True |
| Okiru-Attack | botnet |  | iot23 | 2 | True |
| BFA | bruteforce |  | insdn | 782 | True |
| FTP-Patator | bruteforce |  | cicids2017 | 183 | True |
| password | bruteforce |  | toniot | 14618 | True |
| SSH-Bruteforce | bruteforce |  | cicids2018 | 1955 | True |
| SSH-Patator | bruteforce |  | cicids2017 | 142 | True |
| ddos | ddos |  | toniot | 11578 | True |
| DDoS | ddos |  | cicids2017, insdn, iot23 | 6990 | True |
| DDOS attack-HOIC | ddos |  | cicids2018 | 4210 | True |
| DDOS attack-LOIC-UDP | ddos |  | cicids2018 | 42 | True |
| dos | dos |  | toniot | 5097 | True |
| DoS | dos |  | insdn, unswnb15 | 32224 | True |
| DoS attacks-GoldenEye | dos |  | cicids2018 | 826 | True |
| DoS attacks-Hulk | dos |  | cicids2018 | 2522 | True |
| DoS attacks-SlowHTTPTest | dos |  | cicids2018 | 2 | True |
| DoS attacks-Slowloris | dos |  | cicids2018 | 145 | True |
| DoS GoldenEye | dos |  | cicids2017 | 474 | True |
| DoS Hulk | dos |  | cicids2017 | 7898 | True |
| DoS Slowhttptest | dos |  | cicids2017 | 160 | True |
| DoS slowloris | dos |  | cicids2017 | 172 | True |
| Hulk-Evasive | dos | Evasive | usbids1 | 17499 | True |
| Hulk-NoDefense | dos | NoDefense | usbids1 | 19467 | True |
| Hulk-Reqtimeout | dos | Reqtimeout | usbids1 | 19612 | True |
| Hulk-Security2 | dos | Security2 | usbids1 | 18691 | True |
| Slowhttptest-Evasive | dos | Evasive | usbids1 | 130 | True |
| Slowhttptest-NoDefense | dos | NoDefense | usbids1 | 140 | True |
| Slowhttptest-Reqtimeout | dos | Reqtimeout | usbids1 | 112 | True |
| Slowhttptest-Security2 | dos | Security2 | usbids1 | 136 | True |
| Slowloris-Evasive | dos | Evasive | usbids1 | 19 | True |
| Slowloris-NoDefense | dos | NoDefense | usbids1 | 23 | True |
| Slowloris-Reqtimeout | dos | Reqtimeout | usbids1 | 151 | True |
| Slowloris-Security2 | dos | Security2 | usbids1 | 18 | True |
| TCPFlood-Evasive | dos | Evasive | usbids1 | 4366 | True |
| TCPFlood-NoDefense | dos | NoDefense | usbids1 | 4080 | True |
| TCPFlood-Reqtimeout | dos | Reqtimeout | usbids1 | 4374 | True |
| TCPFlood-Security2 | dos | Security2 | usbids1 | 4221 | True |
| Backdoor | exploit_malware |  | unswnb15 | 323 | True |
| backdoor | exploit_malware |  | toniot | 1812 | True |
| Exploits | exploit_malware |  | unswnb15 | 18400 | True |
| Infilteration | exploit_malware |  | cicids2018 | 2317 | True |
| Infiltration | exploit_malware |  | cicids2017 | 2 | True |
| Shellcode | exploit_malware |  | unswnb15 | 877 | True |
| U2R | exploit_malware |  | insdn | 8 | True |
| Worms | exploit_malware |  | unswnb15 | 116 | True |
| Analysis | other |  | unswnb15 | 391 | True |
| Attack | other |  | iot23 | 537 | True |
| Fuzzers | other |  | unswnb15 | 12580 | True |
| Generic | other |  | unswnb15 | 583 | True |
| mitm | other |  | toniot | 993 | True |
| PartOfAHorizontalPortScan | scan |  | iot23 | 12764 | True |
| PortScan | scan |  | cicids2017 | 84 | True |
| Probe | scan |  | insdn | 32338 | True |
| Reconnaissance | scan |  | unswnb15 | 5132 | True |
| scanning | scan |  | toniot | 4076 | True |
| Brute Force -Web | web |  | cicids2018 | 9 | True |
| Brute Force -XSS | web |  | cicids2018 | 2 | False |
| injection | web |  | toniot | 19707 | True |
| SQL Injection | web |  | cicids2018 | 1 | False |
| Web Attack � Brute Force | web |  | cicids2017 | 61 | True |
| Web Attack � Sql Injection | web |  | cicids2017 | 1 | False |
| Web Attack � XSS | web |  | cicids2017 | 29 | True |
| Web-Attack | web |  | insdn | 110 | True |
| xss | web |  | toniot | 8338 | True |
