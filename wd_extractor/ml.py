#!/usr/bin/env python
import sys
from pathlib import Path
import os
import subprocess
import configparser
from os.path import exists
from sec import SECApi
import json
from pycorenlp import StanfordCoreNLP
import collections
from flatten_json import flatten


def main():
    msg = ''
    details = ''
    vps = {}
    avps = []
    stocks = []
    alerts = []
    alerted = []
    instrades = set()

    nlp = StanfordCoreNLP('http://localhost:9000')

    sec = SECApi('insiders','form4')

    pathlist = Path("../data/2017").glob('**/*.txt')
    for path in pathlist:
        handle = open(path, "r")
        text = handle.read()
        #print(text)
        output_text = sec.form4tojson(text)
        output_dict = json.loads(output_text)
        output_dict = flatten(output_dict)
        annotated_sentences = {}
        for key, value in output_dict.items():
            if value is not None:
                annotated_sentences[key] = nlp.annotate(value, properties={
                    'annotators': 'tokenize,ssplit,ner',
                    'outputFormat': 'json'
                })




    #myapi = SECApi('insiders', 'form4');
            
    #j = json.loads(myapi.form4tojson('https://www.sec.gov/Archives/edgar/data/1633978/000163397817000072/0001633978-17-000072.txt'));
    '''s = j['ownershipdocument']['issuer']['issuertradingsymbol'].upper();
    vps['symbol'] = s;
    
    nitems = j.get('ownershipdocument').get('nonderivativetable');
    items = j.get('ownershipdocument').get('derivativetable');
    
    if items:
        t = items.get('derivativetransaction');
        
        if t:
            for i in t:
                if i == 'transactionamounts':
                        details += t['transactiondate']['value']+' ' + t['securitytitle']['value']+' ' +t[i]['transactionshares']['value'] +' code: '+ t[i]['transactionacquireddisposedcode']['value'] + ', price: '+ t[i]['transactionpricepershare']['value'] + ' Shares left: ' + t['posttransactionamounts']['sharesownedfollowingtransaction']['value']+'\n';
                for ii in i:
                    if ii == 'transactionamounts':
                        details +=  i['transactiondate']['value']+' ' +i['securitytitle']['value']+' ' + i[ii]['transactionshares']['value'] + ' code: '+ i[ii]['transactionacquireddisposedcode']['value'] + ', price: '+ i[ii]['transactionpricepershare']['value'] + ' Shares left: ' + i['posttransactionamounts']['sharesownedfollowingtransaction']['value']+'\n';

    if nitems:
        t = nitems.get('nonderivativetransaction');
        if t:
            for i in t:
                if i == 'transactionamounts':
                        details += t['transactiondate']['value']+' ' + t['securitytitle']['value']+' ' +t[i]['transactionshares']['value'] +' code: '+ t[i]['transactionacquireddisposedcode']['value'] + ', price: '+ t[i]['transactionpricepershare']['value'] + ' Shares left: ' + t['posttransactionamounts']['sharesownedfollowingtransaction']['value']+'\n';
                for ii in i:
                    if ii == 'transactionamounts':
                        details +=  i['transactiondate']['value']+' ' +i['securitytitle']['value']+' ' + i[ii]['transactionshares']['value'] + ' code: '+ i[ii]['transactionacquireddisposedcode']['value'] + ', price: '+ i[ii]['transactionpricepershare']['value'] + ' Shares left: ' + i['posttransactionamounts']['sharesownedfollowingtransaction']['value']+'\n';
    
    '''            
if __name__ == '__main__':
        main()
        sys.exit()
