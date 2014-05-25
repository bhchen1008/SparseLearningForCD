'''
Created on 2014/3/24

@author: bhchen
'''
import os,sys
input_f1 = open(sys.argv[1])
output_f = open(sys.argv[2],'w')
lines = input_f1.readlines()
replace_1 = 'tcp,icmp,udp'
replace_2 = 'http,smtp,finger,domain_u,auth,telnet,ftp,eco_i,ntp_u,ecr_i,other,private,pop_3,ftp_data,rje,time,mtp,link,remote_job,gopher,ssh,name,whois,login,imap4,daytime,ctf,nntp,shell,IRC,nnsp,http_443,exec,printer,efs,courier,uucp,klogin,kshell,echo,discard,systat,supdup,iso_tsap,hostnames,csnet_ns,pop_2,sunrpc,uucp_path,netbios_ns,netbios_ssn,netbios_dgm,sql_net,vmnet,bgp,Z39_50,ldap,netstat,urh_i,X11,urp_i,pm_dump,tftp_u,tim_i,red_i'
replace_3 = 'SF,S1,REJ,S2,S0,S3,RSTO,RSTR,RSTOS0,OTH,SH'
replace_4 = 'back.,buffer_overflow.,ftp_write.,guess_passwd.,imap.,ipsweep.,land.,loadmodule.,multihop.,neptune.,nmap.,perl.,phf.,pod.,portsweep.,rootkit.,satan.,smurf.,spy.,teardrop.,warezclient.,warezmaster.'
dict_1 = {}
dict_2 = {}
dict_3 = {}
dict_4 = {}
list_1 = replace_1.split(',')
list_2 = replace_2.split(',')
list_3 = replace_3.split(',')
list_4 = replace_4.split(',')
for j in range(len(list_1)):
    dict_1[list_1[j]] = str(j+1)
for j in range(len(list_2)):
    dict_2[list_2[j]] = str(j+1)
for j in range(len(list_3)):
    dict_3[list_3[j]] = str(j+1)
for j in range(len(list_4)):
    dict_4[list_4[j]] = str(j+1)    

for line in lines:
    splitByComma = line.split(',')
    for j in range(len(splitByComma)):
        if splitByComma[j] in dict_1:
            output_f.write(dict_1[splitByComma[j]])
        elif splitByComma[j] in dict_2:
            output_f.write(dict_2[splitByComma[j]])
        elif splitByComma[j] in dict_3:
            output_f.write(dict_3[splitByComma[j]])
        elif splitByComma[j].strip() in dict_4:
            output_f.write('abnormal.\n')
        else:
            output_f.write(splitByComma[j])
        if j != len(splitByComma)-1:
            output_f.write(',')
#        else:
#            output_f.write('\n')
            
