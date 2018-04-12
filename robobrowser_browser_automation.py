from robobrowser import RoboBrowser
br = RoboBrowser()
br.open("https://<url>")
form = br.get_form()
form['username'] = "user_id"
form['password'] = "password"
br.submit_form(form)
print(str(br.parsed))
