hugo -F --cleanDestinationDir
cd public
git add ./
git commit -m "update MyLog"
git push -u origin master
