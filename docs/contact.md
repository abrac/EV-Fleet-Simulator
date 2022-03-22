---
title: Contact
---

Welcome to our EV-Fleet-Sim community! You can join our community's group chat:
[https://matrix.to/#/#ev-fleet-sim:matrix.org](https://matrix.to/#/#ev-fleet-sim:matrix.org).

For private communcation, you can reach me on my e-mail address: 
`chris <abraham-without-the-A's> [at] gmail [dot] com` or via Matrix:
[https://matrix.to/#/@abrac:matrix.org](https://matrix.to/#/@abrac:matrix.org).

Also, follow the latest news surrounding EV-Fleet-Sim, by following [our RSS feed]({{site.basurl}}/feed.xml) in your favourite RSS news reader.

<h2>Team</h2>

<ul>
  {% for author in site.authors %}
    <li>
        <h2><a href="{{ author.url}} ">{{ author.name }}</a></h2>
        <h3>{{ author.position }}</h3>
        <p>{{ author.content | markdownify }}</p>
    </li>
  {% endfor %}
</ul>
