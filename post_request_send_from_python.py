import requests
headers_params = {'abc':'axy'}
headers_params['login'] = 'login'
headers_params['password'] = 'password'
query =  {
                "event_name": "name_of_event",
                "from": 20180131,
                "to": 20180229
         }
url_profile="https://abc.com/prof.json"
r = requests.post(url_profile+"?batch_size=5000", data=json.dumps(query), headers=headers_params)
print(r.json())
